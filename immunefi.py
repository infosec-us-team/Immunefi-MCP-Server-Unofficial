"""
Immunefi MCP Server using mcp[cli]
An MCP server that provides access to Immunefi bug bounty program information
"""
import asyncio
import json
import logging
from typing import Optional, List, Dict, Any
from mcp.server.fastmcp import FastMCP
import httpx
from pydantic import BaseModel
from datetime import datetime, timedelta


# Configure logging (use stderr to avoid corrupting STDIO transport)
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# Initialize FastMCP server
mcp = FastMCP("immunefi")

# Unified cache for bounties data (6 hours)
bounties_cache = None
bounties_cache_time = None
CACHE_DURATION = 21600  # 6 hours in seconds (6 * 60 * 60)


def validate_project_ids_alphanumeric(project_ids: List[str]) -> None:
    """
    Validates that all project IDs are alphanumeric.
    """
    for project_id in project_ids:
        if not project_id.replace("-","").isalnum():
            logger.warning(f"Invalid project ID format: {project_id}. Must be alphanumeric.")
            raise ValueError(f"Invalid project ID format: '{project_id}'. Project IDs must be alphanumeric.")


# Helper function to fetch all bounties data with caching
async def get_bounties_data() -> List[Dict[str, Any]]:
    """Fetch all bounties data from Immunefi API with 6-hour caching"""
    global bounties_cache, bounties_cache_time
    
    current_time = asyncio.get_event_loop().time()
    
    # Check if cache is still valid
    if bounties_cache is not None and bounties_cache_time is not None:
        if current_time - bounties_cache_time < CACHE_DURATION:
            logger.info("Using cached bounties data")
            return bounties_cache
    
    logger.info("Fetching fresh bounties data from Immunefi API")
    
    # Fetch from API
    async with httpx.AsyncClient(timeout=15.0) as client:
        try:
            response = await client.get("https://immunefi.com/public-api/bounties.json")
            response.raise_for_status()
            bounties = response.json()
            
            # Validate that we received a list
            if not isinstance(bounties, list):
                raise ValueError("Invalid response format from Immunefi API")
                
            bounties_cache = bounties
            bounties_cache_time = current_time
            logger.info(f"Successfully fetched {len(bounties)} bounties from Immunefi API")
            return bounties
        except httpx.TimeoutException:
            raise ValueError("Timeout while fetching bounties data from Immunefi API")
        except httpx.HTTPStatusError as e:
            logger.error(f"HTTP error while fetching bounties: {e.response.status_code}")
            raise ValueError(f"Error from Immunefi API: {e.response.status_code}")
        except httpx.RequestError as e:
            logger.error(f"Request error while fetching bounties: {str(e)}")
            raise ValueError("Failed to connect to Immunefi API")
        except json.JSONDecodeError:
            raise ValueError("Invalid JSON response from Immunefi API")


# Helper function to get project data from cached bounties
async def get_project_data(project_id: str) -> Dict[str, Any]:
    """Get project data from cached bounties data"""
    bounties = await get_bounties_data()
    
    # Find the project by ID or slug
    for bounty in bounties:
        if bounty.get("id") == project_id or bounty.get("slug") == project_id:
            logger.info(f"Found project data for: {project_id}")
            return bounty
    
    logger.warning(f"Project not found: {project_id}")
    raise ValueError(f"Project with ID '{project_id}' not found in Immunefi database")


# Tool to list all Immunefi programs
@mcp.tool()
async def search_program(query: str) -> str:
    """
    Return a list of Immunefi bug bounty programs filtered by search query.
    The search is performed across project name, id, slug, and tags fields.
    Results are limited to a maximum of 20 programs.
    
    Returns: JSON string containing an object with:
        - result: Array of objects, each containing:
            - id: Project ID (string)
        - total_matching: Total number of matching programs (number)
        - returned: Number of programs returned (number, max 20)
    
    Args:
        query: Search query to filter bug bounty programs
    """
    logger.info(f"Listing programs with query: {query}")
    
    try:
        bounties = await get_bounties_data()
        
        # Filter bounties based on query if provided
        if query:
            query_lower = query.lower()
            filtered_bounties = []
            for b in bounties:
                # Search in project name, id, slug, and tags
                if query_lower in b.get("project", "").lower() or \
                   query_lower in b.get("id", "").lower() or \
                   any(query_lower in str(p).lower() for p in b.get("programType") or []) or \
                   any(query_lower in str(p).lower() for p in b.get("productType") or []) or \
                   any(query_lower in str(p).lower() for p in b.get("ecosystem") or []) or \
                   any(query_lower in str(p).lower() for p in b.get("language") or []) or \
                   query_lower in b.get("slug", "").lower():
                    filtered_bounties.append(b)
                    continue
            
            bounties = filtered_bounties
            logger.info(f"Found {len(bounties)} programs matching query: {query}")
        
        # Return only the key information for each program (max 20)
        result = []
        for b in bounties[:20]:  # Limit to 20 programs
            result.append({
                "id": b.get("id") or b.get("slug"),
            })
        
        return json.dumps({
            "result": result,
            "total_matching": len(bounties),
            "returned": len(result)
        })
    except Exception as e:
        logger.error(f"Error in list_programs: {str(e)}")
        return json.dumps({"error": str(e)})


# Tool to get assets in scope for specific programs
@mcp.tool()
async def get_program_assets(project_ids: List[str]) -> str:
    """
    Return assets in scope for specific Immunefi bug bounty programs.
    
    Returns: JSON string containing an object with:
        - result: Array of objects, each containing:
            - project_id: The requested project ID (string)
            - assets: Array of asset objects in scope (array)
            - error: Error message if project not found (string, optional)
    
    Args:
        project_ids: The unique IDs of the projects to retrieve assets for
    """
    logger.info(f"Getting assets for projects: {project_ids}")
    
    if not project_ids:
        raise ValueError("project_ids is required and cannot be empty")
    
    # Validate project IDs are alphanumeric
    validate_project_ids_alphanumeric(project_ids)
    
    # Verify all projects exist
    bounties = await get_bounties_data()
    valid_projects = {b.get("id") or b.get("slug") for b in bounties}
    invalid_projects = [pid for pid in project_ids if pid not in valid_projects]
    
    if invalid_projects:
        logger.warning(f"Projects not found: {invalid_projects}")
        raise ValueError(f"Projects with IDs '{invalid_projects}' not found in Immunefi database")
    
    results = []
    for project_id in project_ids:
        try:
            project_data = await get_project_data(project_id)
            assets = project_data.get("assets", [])
            project_name = project_data.get("project")
            
            logger.info(f"Successfully retrieved {len(assets)} assets for project: {project_id}")
            results.append({
                "project_id": project_id,
                "assets": assets,
            })
        except ValueError as e:
            logger.warning(f"Project not found: {project_id}")
            results.append({
                "project_id": project_id,
                "error": str(e)
            })
        except Exception as e:
            logger.error(f"Unexpected error in get_program_assets for project {project_id}: {str(e)}")
            results.append({
                "project_id": project_id,
                "error": "Internal server error"
            })

    logger.info(f"Successfully retrieved assets for {len(results)} projects")
    return json.dumps({"result": results})


# Tool to get max bounty for programs
@mcp.tool()
async def get_max_bounty(project_ids: List[str]) -> str:
    """
    Return the maximum bounty amount for specific programs.
    
    Returns: JSON string containing an object with:
        - result: Array of objects, each containing:
            - project_id: The requested project ID (string)
            - max_bounty: Maximum bounty amount (number)
            - error: Error message if project not found (string, optional)
    
    Args:
        project_ids: The unique IDs of the projects to retrieve max bounty for
    """
    logger.info(f"Getting max bounty for projects: {project_ids}")
    
    if not project_ids:
        raise ValueError("project_ids is required and cannot be empty")
    
    # Validate project IDs are alphanumeric
    validate_project_ids_alphanumeric(project_ids)
    
    # Verify all projects exist
    bounties = await get_bounties_data()
    valid_projects = {b.get("id") or b.get("slug") for b in bounties}
    invalid_projects = [pid for pid in project_ids if pid not in valid_projects]
    
    if invalid_projects:
        logger.warning(f"Projects not found: {invalid_projects}")
        raise ValueError(f"Projects with IDs '{invalid_projects}' not found in Immunefi database")
    
    results = []
    for project_id in project_ids:
        try:
            project_data = await get_project_data(project_id)
            max_bounty = project_data.get("maxBounty")
            
            results.append({
                "project_id": project_id,
                "max_bounty": max_bounty
            })
        except ValueError as e:
            logger.warning(f"Project not found: {project_id}")
            results.append({
                "project_id": project_id,
                "error": f"Project with ID '{project_id}' not found in Immunefi database"
            })
        except Exception as e:
            logger.error(f"Unexpected error in get_max_bounty for project {project_id}: {str(e)}")
            results.append({
                "project_id": project_id,
                "error": f"Internal server error retrieving max bounty for project '{project_id}'"
            })

    logger.info(f"Successfully retrieved max bounty for {len(results)} projects")
    return json.dumps({"result": results})


# Tool to get launch date for programs
@mcp.tool()
async def get_launch_date(project_ids: List[str]) -> str:
    """
    Return the launch date for specific programs.
    
    Returns: JSON string containing an object with:
        - result: Array of objects, each containing:
            - project_id: The requested project ID (string)
            - launch_date: Launch date timestamp or date string (string/number)
            - error: Error message if project not found (string, optional)
    
    Args:
        project_ids: The unique IDs of the projects to retrieve launch date for
    """
    logger.info(f"Getting launch date for projects: {project_ids}")
    
    if not project_ids:
        raise ValueError("project_ids is required and cannot be empty")
    
    # Validate project IDs are alphanumeric
    validate_project_ids_alphanumeric(project_ids)
    
    # Verify all projects exist
    bounties = await get_bounties_data()
    valid_projects = {b.get("id") or b.get("slug") for b in bounties}
    invalid_projects = [pid for pid in project_ids if pid not in valid_projects]
    
    if invalid_projects:
        logger.warning(f"Projects not found: {invalid_projects}")
        raise ValueError(f"Projects with IDs '{invalid_projects}' not found in Immunefi database")
    
    results = []
    for project_id in project_ids:
        try:
            project_data = await get_project_data(project_id)
            launch_date = project_data.get("launchDate")
            
            results.append({
                "project_id": project_id,
                "launch_date": launch_date
            })
        except ValueError as e:
            logger.warning(f"Project not found: {project_id}")
            results.append({
                "project_id": project_id,
                "error": f"Project with ID '{project_id}' not found in Immunefi database"
            })
        except Exception as e:
            logger.error(f"Unexpected error in get_launch_date for project {project_id}: {str(e)}")
            results.append({
                "project_id": project_id,
                "error": f"Internal server error retrieving launch date for project '{project_id}'"
            })

    logger.info(f"Successfully retrieved launch date for {len(results)} projects")
    return json.dumps({"result": results})


# Tool to get updated date for programs
@mcp.tool()
async def get_updated_date(project_ids: List[str]) -> str:
    """
    Return the updated date for specific programs.
    
    Returns: JSON string containing an object with:
        - result: Array of objects, each containing:
            - project_id: The requested project ID (string)
            - updated_date: Last updated date timestamp or date string (string/number)
            - error: Error message if project not found (string, optional)
    
    Args:
        project_ids: The unique IDs of the projects to retrieve updated date for
    """
    logger.info(f"Getting updated date for projects: {project_ids}")
    
    if not project_ids:
        raise ValueError("project_ids is required and cannot be empty")
    
    # Validate project IDs are alphanumeric
    validate_project_ids_alphanumeric(project_ids)
    
    # Verify all projects exist
    bounties = await get_bounties_data()
    valid_projects = {b.get("id") or b.get("slug") for b in bounties}
    invalid_projects = [pid for pid in project_ids if pid not in valid_projects]
    
    if invalid_projects:
        logger.warning(f"Projects not found: {invalid_projects}")
        raise ValueError(f"Projects with IDs '{invalid_projects}' not found in Immunefi database")
    
    results = []
    for project_id in project_ids:
        try:
            project_data = await get_project_data(project_id)
            updated_date = project_data.get("updatedDate")
            
            results.append({
                "project_id": project_id,
                "updated_date": updated_date
            })
        except ValueError as e:
            logger.warning(f"Project not found: {project_id}")
            results.append({
                "project_id": project_id,
                "error": f"Project with ID '{project_id}' not found in Immunefi database"
            })
        except Exception as e:
            logger.error(f"Unexpected error in get_updated_date for project {project_id}: {str(e)}")
            results.append({
                "project_id": project_id,
                "error": f"Internal server error retrieving updated date for project '{project_id}'"
            })

    logger.info(f"Successfully retrieved updated date for {len(results)} projects")
    return json.dumps({"result": results})


# Tool to get KYC requirement for programs
@mcp.tool()
async def is_kyc_required(project_ids: List[str]) -> str:
    """
    Return the KYC requirement for specific programs.
    
    Returns: JSON string containing an object with:
        - result: Array of objects, each containing:
            - project_id: The requested project ID (string)
            - kyc_required: Whether KYC is required (boolean)
            - error: Error message if project not found (string, optional)
    
    Args:
        project_ids: The unique IDs of the projects to retrieve KYC status for
    """
    logger.info(f"Getting KYC requirement for projects: {project_ids}")
    
    if not project_ids:
        raise ValueError("project_ids is required and cannot be empty")
    
    # Validate project IDs are alphanumeric
    validate_project_ids_alphanumeric(project_ids)
    
    # Verify all projects exist
    bounties = await get_bounties_data()
    valid_projects = {b.get("id") or b.get("slug") for b in bounties}
    invalid_projects = [pid for pid in project_ids if pid not in valid_projects]
    
    if invalid_projects:
        logger.warning(f"Projects not found: {invalid_projects}")
        raise ValueError(f"Projects with IDs '{invalid_projects}' not found in Immunefi database")
    
    results = []
    for project_id in project_ids:
        try:
            project_data = await get_project_data(project_id)
            kyc = project_data.get("kyc")
            
            results.append({
                "project_id": project_id,
                "kyc_required": kyc
            })
        except ValueError as e:
            logger.warning(f"Project not found: {project_id}")
            results.append({
                "project_id": project_id,
                "error": f"Project with ID '{project_id}' not found in Immunefi database"
            })
        except Exception as e:
            logger.error(f"Unexpected error in get_kyc for project {project_id}: {str(e)}")
            results.append({
                "project_id": project_id,
                "error": f"Internal server error retrieving KYC requirement for project '{project_id}'"
            })

    logger.info(f"Successfully retrieved KYC requirement for {len(results)} projects")
    return json.dumps({"result": results})


# Tool to get rewards for programs
@mcp.tool()
async def get_rewards(project_ids: List[str]) -> str:
    """
    Return the rewards information for specific programs.
    
    Returns: JSON string containing an object with:
        - result: Array of objects, each containing:
            - project_id: The requested project ID (string)
            - rewards: Rewards structure with severity levels and amounts (object/array)
            - error: Error message if project not found (string, optional)
    
    Args:
        project_ids: The unique IDs of the projects to retrieve rewards for
    """
    logger.info(f"Getting rewards for projects: {project_ids}")
    
    if not project_ids:
        raise ValueError("project_ids is required and cannot be empty")
    
    # Validate project IDs are alphanumeric
    validate_project_ids_alphanumeric(project_ids)
    
    # Verify all projects exist
    bounties = await get_bounties_data()
    valid_projects = {b.get("id") or b.get("slug") for b in bounties}
    invalid_projects = [pid for pid in project_ids if pid not in valid_projects]
    
    if invalid_projects:
        logger.warning(f"Projects not found: {invalid_projects}")
        raise ValueError(f"Projects with IDs '{invalid_projects}' not found in Immunefi database")
    
    results = []
    for project_id in project_ids:
        try:
            project_data = await get_project_data(project_id)
            rewards = project_data.get("rewards")
            
            results.append({
                "project_id": project_id,
                "rewards": rewards
            })
        except ValueError as e:
            logger.warning(f"Project not found: {project_id}")
            results.append({
                "project_id": project_id,
                "error": f"Project with ID '{project_id}' not found in Immunefi database"
            })
        except Exception as e:
            logger.error(f"Unexpected error in get_rewards for project {project_id}: {str(e)}")
            results.append({
                "project_id": project_id,
                "error": f"Internal server error retrieving rewards for project '{project_id}'"
            })

    logger.info(f"Successfully retrieved rewards for {len(results)} projects")
    return json.dumps({"result": results})


# Tool to get impacts for programs
@mcp.tool()
async def get_impacts(project_ids: List[str]) -> str:
    """
    Return the impacts information for specific programs.
    
    Returns: JSON string containing an object with:
        - result: Array of objects, each containing:
            - project_id: The requested project ID (string)
            - impacts: Impact categories and descriptions (object/array)
            - error: Error message if project not found (string, optional)
    
    Args:
        project_ids: The unique IDs of the projects to retrieve impacts for
    """
    logger.info(f"Getting impacts for projects: {project_ids}")
    
    if not project_ids:
        raise ValueError("project_ids is required and cannot be empty")
    
    # Validate project IDs are alphanumeric
    validate_project_ids_alphanumeric(project_ids)
    
    # Verify all projects exist
    bounties = await get_bounties_data()
    valid_projects = {b.get("id") or b.get("slug") for b in bounties}
    invalid_projects = [pid for pid in project_ids if pid not in valid_projects]
    
    if invalid_projects:
        logger.warning(f"Projects not found: {invalid_projects}")
        raise ValueError(f"Projects with IDs '{invalid_projects}' not found in Immunefi database")
    
    results = []
    for project_id in project_ids:
        try:
            project_data = await get_project_data(project_id)
            impacts = project_data.get("impacts")
            
            results.append({
                "project_id": project_id,
                "impacts": impacts
            })
        except ValueError as e:
            logger.warning(f"Project not found: {project_id}")
            results.append({
                "project_id": project_id,
                "error": f"Project with ID '{project_id}' not found in Immunefi database"
            })
        except Exception as e:
            logger.error(f"Unexpected error in get_impacts for project {project_id}: {str(e)}")
            results.append({
                "project_id": project_id,
                "error": f"Internal server error retrieving impacts for project '{project_id}'"
            })

    logger.info(f"Successfully retrieved impacts for {len(results)} projects")
    return json.dumps({"result": results})


# Tool to get specific tags for programs
@mcp.tool()
async def get_tags(project_ids: List[str]) -> str:
    """
    Return the tags information for specific programs (productType, ecosystem, programType, language).
    
    Returns: JSON string containing an object with:
        - result: Array of objects, each containing:
            - project_id: The requested project ID (string)
            - tags: Object with tag categories (productType, ecosystem, programType, language), each containing an array of strings
            - error: Error message if project not found (string, optional)
    
    Args:
        project_ids: The unique IDs of the projects to retrieve tags for
    """
    logger.info(f"Getting tags for projects: {project_ids}")
    
    if not project_ids:
        raise ValueError("project_ids is required and cannot be empty")
    
    # Validate project IDs are alphanumeric
    validate_project_ids_alphanumeric(project_ids)
    
    # Verify all projects exist
    bounties = await get_bounties_data()
    valid_projects = {b.get("id") or b.get("slug") for b in bounties}
    invalid_projects = [pid for pid in project_ids if pid not in valid_projects]
    
    if invalid_projects:
        logger.warning(f"Projects not found: {invalid_projects}")
        raise ValueError(f"Projects with IDs '{invalid_projects}' not found in Immunefi database")
    
    results = []
    for project_id in project_ids:
        try:
            project_data = await get_project_data(project_id)
            
            # Extract tags from the project data
            tags = {}
            tags_to_extract = ["productType", "ecosystem", "programType", "language"]
            
            for tag_type in tags_to_extract:
                tags[tag_type] = project_data.get(tag_type, [])
            
            results.append({
                "project_id": project_id,
                "tags": tags
            })
        except ValueError as e:
            logger.warning(f"Project not found: {project_id}")
            results.append({
                "project_id": project_id,
                "error": f"Project with ID '{project_id}' not found in Immunefi database"
            })
        except Exception as e:
            logger.error(f"Unexpected error in get_tags for project {project_id}: {str(e)}")
            results.append({
                "project_id": project_id,
                "error": f"Internal server error retrieving tags for project '{project_id}'"
            })

    logger.info(f"Successfully retrieved tags for {len(results)} projects")
    return json.dumps({"result": results})


# Tool to filter project IDs by language tag
@mcp.tool()
async def filter_by_language(project_ids: List[str], language: str) -> str:
    """
    Return the subset of provided project_ids whose "language" field contains the given language.
    Matching is case-insensitive.

    Returns: JSON string containing an object with:
        - result: Object containing:
            - field: "language"
            - value: The language filter applied (string)
            - matching_programs: Array of objects, each containing:
                - id: Project ID (string)
                - language: The program's language array (array)
            - count: Number of matching programs (number)

    Args:
        project_ids: List of project IDs to filter
        language: Language value to match (case-insensitive)
    """
    logger.info(f"Filtering by language='{language}' for projects: {project_ids}")

    if not project_ids:
        raise ValueError("project_ids is required and cannot be empty")

    if not language or not isinstance(language, str):
        raise ValueError("language is required and must be a non-empty string")

    # Validate project IDs are alphanumeric
    validate_project_ids_alphanumeric(project_ids)

    # Verify all projects exist
    bounties = await get_bounties_data()
    valid_projects = {b.get("id") or b.get("slug") for b in bounties}
    invalid_projects = [pid for pid in project_ids if pid not in valid_projects]

    if invalid_projects:
        logger.warning(f"Projects not found: {invalid_projects}")
        raise ValueError(f"Projects with IDs '{invalid_projects}' not found in Immunefi database")

    language_lower = language.strip().lower()
    matching_programs = []

    for project_id in project_ids:
        try:
            project_data = await get_project_data(project_id)
            langs = project_data.get("language") or []
            # Normalize to list
            if isinstance(langs, str):
                langs_list = [langs]
            elif isinstance(langs, list):
                langs_list = [str(x) for x in langs]
            else:
                langs_list = []

            # Case-insensitive membership check
            langs_lower = {str(x).strip().lower() for x in langs_list}
            if language_lower in langs_lower:
                matching_programs.append({
                    "id": project_id,
                    "language": langs_list
                })
        except Exception as e:
            logger.error(f"Unexpected error filtering language for project {project_id}: {str(e)}")
            continue

    logger.info(f"Found {len(matching_programs)} programs matching language '{language}'")
    return json.dumps({
        "result": {
            "field": "language",
            "value": language,
            "matching_programs": matching_programs,
            "count": len(matching_programs)
        }
    })


# Tool to filter project IDs by ecosystem tag
@mcp.tool()
async def filter_by_ecosystem(project_ids: List[str], ecosystem: str) -> str:
    """
    Return the subset of provided project_ids whose "ecosystem" field contains the given ecosystem.
    Matching is case-insensitive.

    Returns: JSON string containing an object with:
        - result: Object containing:
            - field: "ecosystem"
            - value: The ecosystem filter applied (string)
            - matching_programs: Array of objects, each containing:
                - id: Project ID (string)
                - ecosystem: The program's ecosystem array (array)
            - count: Number of matching programs (number)

    Args:
        project_ids: List of project IDs to filter
        ecosystem: Ecosystem value to match (case-insensitive)
    """
    logger.info(f"Filtering by ecosystem='{ecosystem}' for projects: {project_ids}")

    if not project_ids:
        raise ValueError("project_ids is required and cannot be empty")

    if not ecosystem or not isinstance(ecosystem, str):
        raise ValueError("ecosystem is required and must be a non-empty string")

    # Validate project IDs are alphanumeric
    validate_project_ids_alphanumeric(project_ids)

    # Verify all projects exist
    bounties = await get_bounties_data()
    valid_projects = {b.get("id") or b.get("slug") for b in bounties}
    invalid_projects = [pid for pid in project_ids if pid not in valid_projects]

    if invalid_projects:
        logger.warning(f"Projects not found: {invalid_projects}")
        raise ValueError(f"Projects with IDs '{invalid_projects}' not found in Immunefi database")

    ecosystem_lower = ecosystem.strip().lower()
    matching_programs = []

    for project_id in project_ids:
        try:
            project_data = await get_project_data(project_id)
            ecos = project_data.get("ecosystem") or []
            # Normalize to list
            if isinstance(ecos, str):
                ecos_list = [ecos]
            elif isinstance(ecos, list):
                ecos_list = [str(x) for x in ecos]
            else:
                ecos_list = []

            # Case-insensitive membership check
            ecos_lower = {str(x).strip().lower() for x in ecos_list}
            if ecosystem_lower in ecos_lower:
                matching_programs.append({
                    "id": project_id,
                    "ecosystem": ecos_list
                })
        except Exception as e:
            logger.error(f"Unexpected error filtering ecosystem for project {project_id}: {str(e)}")
            continue

    logger.info(f"Found {len(matching_programs)} programs matching ecosystem '{ecosystem}'")
    return json.dumps({
        "result": {
            "field": "ecosystem",
            "value": ecosystem,
            "matching_programs": matching_programs,
            "count": len(matching_programs)
        }
    })


# Tool to search for programs by bounty range
@mcp.tool()
async def filter_by_bounty(min_bounty: int = 0, max_bounty: Optional[int] = None, project_ids: Optional[List[str]] = None) -> str:
    """
    Return the IDs of all bug bounty programs with a maximum bounty within the specified range.
    If project_ids is provided, it works like a filter by only searching within those specific projects.
    
    Returns: JSON string containing an object with:
        - result: Object containing:
            - min_bounty: The minimum bounty filter applied (number)
            - max_bounty: The maximum bounty filter applied (number or null)
            - matching_programs: Array of objects, each containing:
                - id: Project ID (string)
                - max_bounty: Maximum bounty amount (number)
            - count: Number of matching programs (number)
    
    Args:
        min_bounty: The minimum bounty amount to filter programs by (default: 0)
        max_bounty: The maximum bounty amount to filter programs by (optional, no upper limit if not specified)
        project_ids: Optional list of project IDs to limit the search to (default: None, searches all projects)
    """
    logger.info(f"Searching for programs with bounty range: {min_bounty} - {max_bounty if max_bounty else 'unlimited'}. Project IDs filter: {project_ids}")
    
    if min_bounty < 0:
        raise ValueError("min_bounty must be a non-negative integer")
    
    if max_bounty is not None and max_bounty < min_bounty:
        raise ValueError("max_bounty must be greater than or equal to min_bounty")
    
    # Get all bounties data
    all_bounties_data = await get_bounties_data()
    
    # Filter bounties data if project_ids is provided
    if project_ids:
        # Validate project IDs are alphanumeric
        validate_project_ids_alphanumeric(project_ids)
        
        # Verify all projects exist
        valid_projects = {b.get("id") or b.get("slug") for b in all_bounties_data}
        invalid_projects = [pid for pid in project_ids if pid not in valid_projects]
        
        if invalid_projects:
            logger.warning(f"Projects not found: {invalid_projects}")
            raise ValueError(f"Projects with IDs '{invalid_projects}' not found in Immunefi database")
        
        # Filter bounties to only include specified project IDs
        all_bounties_data = [
            program for program in all_bounties_data
            if (program.get("id") or program.get("slug")) in set(project_ids)
        ]
    
    # Find programs with max bounty within the specified range
    matching_programs = []
    for program in all_bounties_data:
        # Check if the program has a max bounty within the range
        program_max_bounty = program.get("maxBounty", 0)  # Default to 0 if not specified
        
        # Check lower bound
        if program_max_bounty < min_bounty:
            continue
        
        # Check upper bound if specified
        if max_bounty is not None and program_max_bounty > max_bounty:
            continue
        
        # Add the program ID to the results
        program_id = program.get("id", program.get("slug", "unknown"))
        matching_programs.append({
            "id": program_id,
            "max_bounty": program_max_bounty
        })
    
    # Sort results by max bounty in descending order
    matching_programs.sort(key=lambda x: x["max_bounty"], reverse=True)
    
    logger.info(f"Found {len(matching_programs)} programs with bounty range: {min_bounty} - {max_bounty if max_bounty else 'unlimited'}")
    return json.dumps({
        "result": {
            "min_bounty": min_bounty,
            "max_bounty": max_bounty,
            "matching_programs": matching_programs,
            "count": len(matching_programs)
        }
    })


# Helper function to extract GitHub repositories from text
def extract_github_repos_from_text(text):
    """
    Extract GitHub repository URLs from text using regex
    """
    import re
    # Pattern to match GitHub URLs (https://github.com/username/repo or similar patterns)
    github_pattern = r'https://github\.com/([a-zA-Z0-9._-]+)/([a-zA-Z0-9._-]+)'
    matches = re.findall(github_pattern, text)
    # Return unique repository URLs
    repos = set()
    for match in matches:
        username, repo = match
        repo_url = f"https://github.com/{username}/{repo}"
        repos.add(repo_url)
    return list(repos)


# Helper function to extract GitHub repositories from a program's data
def extract_github_repos_from_program(program_data):
    """
    Extract all GitHub repository URLs from a program's data
    """
    repos = set()
    
    # Look for URLs in various fields of the program data
    def extract_from_dict(data, path=""):
        if isinstance(data, dict):
            for key, value in data.items():
                new_path = f"{path}.{key}" if path else key
                extract_from_dict(value, new_path)
        elif isinstance(data, list):
            for i, item in enumerate(data):
                new_path = f"{path}[{i}]"
                extract_from_dict(item, new_path)
        elif isinstance(data, str):
            # Extract GitHub repos from string values
            found_repos = extract_github_repos_from_text(data)
            repos.update(found_repos)
    
    extract_from_dict(program_data)
    return list(repos)


# Tool to search for GitHub repositories by protocol ID
@mcp.tool()
async def search_github_repos(project_ids: List[str]) -> str:
    """
    Return all GitHub repositories associated with specific protocols by their IDs.
    
    Returns: JSON string containing an object with:
        - result: Array of objects, each containing:
            - project_id: The requested project ID (string)
            - github_repositories: Array of GitHub repository URLs (array of strings)
            - count: Number of repositories found (number)
            - error: Error message if project not found (string, optional)
    
    Args:
        project_ids: The unique IDs of the projects to search for GitHub repositories
    """
    logger.info(f"Searching for GitHub repositories for projects: {project_ids}")
    
    if not project_ids:
        raise ValueError("project_ids is required and cannot be empty")
    
    # Validate project IDs are alphanumeric
    validate_project_ids_alphanumeric(project_ids)
    
    # Verify all projects exist
    bounties = await get_bounties_data()
    valid_projects = {b.get("id") or b.get("slug") for b in bounties}
    invalid_projects = [pid for pid in project_ids if pid not in valid_projects]
    
    if invalid_projects:
        logger.warning(f"Projects not found: {invalid_projects}")
        raise ValueError(f"Projects with IDs '{invalid_projects}' not found in Immunefi database")
    
    results = []
    for project_id in project_ids:
        try:
            project_data = await get_project_data(project_id)
            github_repos = extract_github_repos_from_program(project_data)
            
            logger.info(f"Found {len(github_repos)} GitHub repositories for project: {project_id}")
            results.append({
                "project_id": project_id,
                "github_repositories": sorted(github_repos),  # Sort alphabetically for consistency
                "count": len(github_repos)
            })
        except ValueError as e:
            logger.warning(f"Project not found: {project_id}")
            results.append({
                "project_id": project_id,
                "error": f"Project with ID '{project_id}' not found in Immunefi database"
            })
        except Exception as e:
            logger.error(f"Unexpected error in search_github_repos for project {project_id}: {str(e)}")
            results.append({
                "project_id": project_id,
                "error": f"Internal server error retrieving GitHub repositories for project '{project_id}'"
            })
    
    logger.info(f"Successfully retrieved GitHub repositories for {len(results)} projects")
    return json.dumps({"result": results})


# Tool to get all project IDs
@mcp.tool()
async def get_all_project_ids() -> str:
    """
    Return all project IDs from the Immunefi bug bounty programs.
    
    Returns: JSON string containing an object with:
        - result: Object containing:
            - project_ids: Array of all project IDs (array of strings)
            - count: Total number of project IDs (number)
        - error: Error message if operation fails (string, optional)
    """
    logger.info("Getting all project IDs")
    
    try:
        bounties = await get_bounties_data()
        
        # Extract all project IDs
        project_ids = []
        for bounty in bounties:
            project_id = bounty.get("id") or bounty.get("slug")
            if project_id:
                project_ids.append(project_id)
        
        logger.info(f"Found {len(project_ids)} project IDs")
        return json.dumps({
            "result": {
                "project_ids": sorted(project_ids),
                "count": len(project_ids)
            }
        })
    except Exception as e:
        logger.error(f"Error in get_all_project_ids: {str(e)}")
        return json.dumps({"error": str(e)})


# Tool to get all available fields from the API
@mcp.tool()
async def get_available_fields() -> str:
    """
    Return all available fields (keys) returned by the Immunefi API for bug bounty programs.
    This shows what data is available for each program.
    
    This is helpful to call before using get_field_values() to know which field names you can request
    for specific programs (e.g., "maxBounty", "kyc", "launchDate", "assets", etc.).
    
    Returns: JSON string containing an object with:
        - result: Object containing:
            - fields: Array of all available field names (array of strings)
            - count: Total number of fields (number)
            - message: Optional message if no data available (string, optional)
        - error: Error message if operation fails (string, optional)
    """
    logger.info("Getting all available fields from API")
    
    try:
        bounties = await get_bounties_data()
        
        if not bounties:
            return json.dumps({
                "result": {
                    "fields": [],
                    "count": 0,
                    "message": "No bounties data available"
                }
            })
        
        # Collect all unique keys from all programs
        all_fields = set()
        for bounty in bounties:
            all_fields.update(bounty.keys())
        
        # Sort fields alphabetically
        sorted_fields = sorted(all_fields)
        
        logger.info(f"Found {len(sorted_fields)} unique fields")
        return json.dumps({
            "result": {
                "fields": sorted_fields,
                "count": len(sorted_fields)
            }
        })
    except Exception as e:
        logger.error(f"Error in get_available_fields: {str(e)}")
        return json.dumps({"error": str(e)})


# Tool to get specific field values for projects
@mcp.tool()
async def get_field_values(project_ids: List[str], field_name: str) -> str:
    """
    Return the value of a specific field for the given project IDs.
    
    Returns: JSON string containing an object with:
        - result: Array of objects, each containing:
            - project_id: The requested project ID (string)
            - field_name: The field name that was queried (string)
            - value: The value of the field (any type)
            - error: Error message if project not found (string, optional)
    
    Args:
        project_ids: List of project IDs to retrieve the field value for
        field_name: The name of the field to retrieve (e.g., "maxBounty", "kyc", "launchDate")
    """
    logger.info(f"Getting field '{field_name}' for projects: {project_ids}")
    
    if not project_ids:
        raise ValueError("project_ids is required and cannot be empty")
    
    if not field_name:
        raise ValueError("field_name is required and cannot be empty")
    
    # Validate project IDs are alphanumeric
    validate_project_ids_alphanumeric(project_ids)
    
    # Verify all projects exist
    bounties = await get_bounties_data()
    valid_projects = {b.get("id") or b.get("slug") for b in bounties}
    invalid_projects = [pid for pid in project_ids if pid not in valid_projects]
    
    if invalid_projects:
        logger.warning(f"Projects not found: {invalid_projects}")
        raise ValueError(f"Projects with IDs '{invalid_projects}' not found in Immunefi database")
    
    results = []
    for project_id in project_ids:
        try:
            project_data = await get_project_data(project_id)
            field_value = project_data.get(field_name)
            
            results.append({
                "project_id": project_id,
                "field_name": field_name,
                "value": field_value
            })
        except ValueError as e:
            logger.warning(f"Project not found: {project_id}")
            results.append({
                "project_id": project_id,
                "field_name": field_name,
                "error": f"Project with ID '{project_id}' not found in Immunefi database"
            })
        except Exception as e:
            logger.error(f"Unexpected error for project {project_id}: {str(e)}")
            results.append({
                "project_id": project_id,
                "field_name": field_name,
                "error": f"Internal server error: {str(e)}"
            })
    
    logger.info(f"Successfully retrieved field '{field_name}' for {len(results)} projects")
    return json.dumps({"result": results})


# Tool to search programs updated in the past X days
@mcp.tool()
async def search_updated_recently(days: Optional[int] = None, months: Optional[int] = None, project_ids: Optional[List[str]] = None) -> str:
    """
    Return project IDs of programs that have been updated in the past X days or months.
    If project_ids is provided, only search within those specific projects.
    You must specify either days or months, but not both.
    
    Returns: JSON string containing an object with:
        - result: Object containing:
            - time_period: Description of the time period searched (string)
            - cutoff_date: ISO format cutoff date (string)
            - matching_programs: Array of objects, each containing:
                - id: Project ID (string)
                - updated_date: Last updated date (string/number)
            - count: Number of matching programs (number)
    
    Args:
        days: Number of days to look back (e.g., 30 for programs updated in the past 30 days)
        months: Number of months to look back (e.g., 2 for programs updated in the past 2 months)
        project_ids: Optional list of project IDs to limit the search to (default: None, searches all projects)
    """
    logger.info(f"Searching for programs updated in the past {days} days or {months} months. Project IDs filter: {project_ids}")
    
    if days is None and months is None:
        raise ValueError("You must specify either 'days' or 'months' parameter")
    
    if days is not None and months is not None:
        raise ValueError("You cannot specify both 'days' and 'months' parameters. Choose one.")
    
    if days is not None and days <= 0:
        raise ValueError("days must be a positive integer")
    
    if months is not None and months <= 0:
        raise ValueError("months must be a positive integer")
    
    # Calculate the cutoff date (in milliseconds timestamp)
    current_time = datetime.now()
    if days is not None:
        cutoff_date = current_time - timedelta(days=days)
        time_period = f"{days} days"
    else:
        # Approximate months as 30 days each
        cutoff_date = current_time - timedelta(days=months * 30)
        time_period = f"{months} months"
    
    cutoff_timestamp = int(cutoff_date.timestamp() * 1000)  # Convert to milliseconds
    
    # Get all bounties data
    bounties = await get_bounties_data()
    
    # Filter bounties data if project_ids is provided
    if project_ids:
        # Validate project IDs are alphanumeric
        validate_project_ids_alphanumeric(project_ids)
        
        # Verify all projects exist
        valid_projects = {b.get("id") or b.get("slug") for b in bounties}
        invalid_projects = [pid for pid in project_ids if pid not in valid_projects]
        
        if invalid_projects:
            logger.warning(f"Projects not found: {invalid_projects}")
            raise ValueError(f"Projects with IDs '{invalid_projects}' not found in Immunefi database")
        
        # Filter bounties to only include specified project IDs
        bounties = [
            program for program in bounties
            if (program.get("id") or program.get("slug")) in set(project_ids)
        ]
    
    # Find programs updated after the cutoff date
    matching_programs = []
    programs_without_date = 0
    parse_errors = 0
    
    for program in bounties:
        updated_date_value = program.get("updatedDate")
        
        if updated_date_value is None:
            programs_without_date += 1
            continue
        
        try:
            # Parse the updated date
            updated_timestamp = None
            
            # Try as integer/float timestamp in milliseconds
            if isinstance(updated_date_value, (int, float)):
                updated_timestamp = int(updated_date_value)
            elif isinstance(updated_date_value, str) and updated_date_value.isdigit():
                updated_timestamp = int(updated_date_value)
            else:
                # Try parsing as ISO date string and convert to timestamp
                try:
                    if isinstance(updated_date_value, str):
                        dt = datetime.fromisoformat(updated_date_value.replace('Z', '+00:00'))
                        updated_timestamp = int(dt.timestamp() * 1000)
                except ValueError:
                    pass
            
            # If we successfully parsed the timestamp and it's after the cutoff
            if updated_timestamp and updated_timestamp >= cutoff_timestamp:
                project_id = program.get("id") or program.get("slug")
                matching_programs.append({
                    "id": project_id,
                    "updated_date": updated_date_value
                })
            elif updated_timestamp is None:
                parse_errors += 1
                logger.debug(f"Could not parse date '{updated_date_value}' (type: {type(updated_date_value).__name__}) for program {program.get('id')}")
                
        except Exception as e:
            parse_errors += 1
            logger.warning(f"Error parsing date for program {program.get('id')}: {str(e)} (date value: {updated_date_value}, type: {type(updated_date_value).__name__})")
            continue
    
    logger.info(f"Programs without updatedDate: {programs_without_date}, Parse errors: {parse_errors}")
    
    # Sort by updated date (most recent first)
    matching_programs.sort(key=lambda x: x.get("updated_date", 0), reverse=True)
    
    logger.info(f"Found {len(matching_programs)} programs updated in the past {time_period}")
    return json.dumps({
        "result": {
            "time_period": time_period,
            "cutoff_date": cutoff_date.isoformat(),
            "matching_programs": matching_programs,
            "count": len(matching_programs)
        }
    })


# Tool to search programs updated after a specific date
@mcp.tool()
async def search_updated_after_date(date: str, project_ids: Optional[List[str]] = None) -> str:
    """
    Return project IDs of programs that have been updated after a specific date.
    If project_ids is provided, only search within those specific projects.
    
    Returns: JSON string containing an object with:
        - result: Object containing:
            - cutoff_date: ISO format cutoff date (string)
            - matching_programs: Array of objects, each containing:
                - id: Project ID (string)
                - updated_date: Last updated date (string/number)
            - count: Number of matching programs (number)
    
    Args:
        date: The cutoff date in ISO format (e.g., "2024-01-01" or "2024-01-01T00:00:00")
        project_ids: Optional list of project IDs to limit the search to (default: None, searches all projects)
    """
    logger.info(f"Searching for programs updated after {date}. Project IDs filter: {project_ids}")
    
    if not date:
        raise ValueError("date parameter is required")
    
    # Parse the cutoff date
    try:
        cutoff_date = datetime.fromisoformat(date.replace('Z', '+00:00'))
    except ValueError:
        raise ValueError(f"Invalid date format: '{date}'. Use ISO format like '2024-01-01' or '2024-01-01T00:00:00'")
    
    cutoff_timestamp = int(cutoff_date.timestamp() * 1000)  # Convert to milliseconds
    
    # Get all bounties data
    bounties = await get_bounties_data()
    
    # Filter bounties data if project_ids is provided
    if project_ids:
        # Validate project IDs are alphanumeric
        validate_project_ids_alphanumeric(project_ids)
        
        # Verify all projects exist
        valid_projects = {b.get("id") or b.get("slug") for b in bounties}
        invalid_projects = [pid for pid in project_ids if pid not in valid_projects]
        
        if invalid_projects:
            logger.warning(f"Projects not found: {invalid_projects}")
            raise ValueError(f"Projects with IDs '{invalid_projects}' not found in Immunefi database")
        
        # Filter bounties to only include specified project IDs
        bounties = [
            program for program in bounties
            if (program.get("id") or program.get("slug")) in set(project_ids)
        ]
    
    # Find programs updated after the cutoff date
    matching_programs = []
    programs_without_date = 0
    parse_errors = 0
    
    for program in bounties:
        updated_date_value = program.get("updatedDate")
        
        if updated_date_value is None:
            programs_without_date += 1
            continue
        
        try:
            # Parse the updated date
            updated_timestamp = None
            
            # Try as integer/float timestamp in milliseconds
            if isinstance(updated_date_value, (int, float)):
                updated_timestamp = int(updated_date_value)
            elif isinstance(updated_date_value, str) and updated_date_value.isdigit():
                updated_timestamp = int(updated_date_value)
            else:
                # Try parsing as ISO date string and convert to timestamp
                try:
                    if isinstance(updated_date_value, str):
                        dt = datetime.fromisoformat(updated_date_value.replace('Z', '+00:00'))
                        updated_timestamp = int(dt.timestamp() * 1000)
                except ValueError:
                    pass
            
            # If we successfully parsed the timestamp and it's after the cutoff
            if updated_timestamp and updated_timestamp >= cutoff_timestamp:
                project_id = program.get("id") or program.get("slug")
                matching_programs.append({
                    "id": project_id,
                    "updated_date": updated_date_value
                })
            elif updated_timestamp is None:
                parse_errors += 1
                logger.debug(f"Could not parse date '{updated_date_value}' (type: {type(updated_date_value).__name__}) for program {program.get('id')}")
                
        except Exception as e:
            parse_errors += 1
            logger.warning(f"Error parsing date for program {program.get('id')}: {str(e)} (date value: {updated_date_value}, type: {type(updated_date_value).__name__})")
            continue
    
    logger.info(f"Programs without updatedDate: {programs_without_date}, Parse errors: {parse_errors}")
    
    # Sort by updated date (most recent first)
    matching_programs.sort(key=lambda x: x.get("updated_date", ""), reverse=True)
    
    logger.info(f"Found {len(matching_programs)} programs updated after {date}")
    return json.dumps({
        "result": {
            "cutoff_date": cutoff_date.isoformat(),
            "matching_programs": matching_programs,
            "count": len(matching_programs)
        }
    })


def main():
    # Run the MCP server with STDIO transport
    mcp.run(transport='stdio')


if __name__ == "__main__":
    main()
