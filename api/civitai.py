# ================================================
# File: api/civitai.py
# ================================================
import requests
import json
from typing import List, Optional, Dict, Any, Union

class CivitaiAPI:
    """Simple wrapper for interacting with the Civitai API v1."""
    DEFAULT_DOMAIN = "civitai.com"
    ALLOWED_DOMAINS = ("civitai.com", "civitai.red")
    MEILI_TOKEN = "8c46eb2508e21db1e9828a97968d91ab1ca1caa5f70a00e88a2ba1e286603b61"

    def __init__(self, api_key: Optional[str] = None, domain: Optional[str] = None):
        self.api_key = api_key
        self.domain = self.normalize_domain(domain)
        self.base_url = f"https://{self.domain}/api/v1"
        self.search_url = "https://search.civitai.com/multi-search"
        self.image_base_url = "https://image.civitai.com/xG1nkqKTMzGDvpLrqFT7QA"
        self.base_headers = {'Content-Type': 'application/json'}
        if api_key:
            self.base_headers["Authorization"] = f"Bearer {api_key}"
            print(f"[Civitai API] Using API Key for {self.domain}.")
        else:
            print(f"[Civitai API] No API Key provided for {self.domain}.")

    @classmethod
    def normalize_domain(cls, domain: Optional[str]) -> str:
        """Return a supported Civitai domain, defaulting safely to civitai.com."""
        value = (domain or cls.DEFAULT_DOMAIN).strip().lower()
        value = value.removeprefix("https://").removeprefix("http://").strip("/")
        return value if value in cls.ALLOWED_DOMAINS else cls.DEFAULT_DOMAIN

    @classmethod
    def opposite_domain(cls, domain: Optional[str]) -> str:
        normalized = cls.normalize_domain(domain)
        return "civitai.red" if normalized == "civitai.com" else "civitai.com"

    def _get_request_headers(self, method: str, has_json_data: bool) -> Dict[str, str]:
        """Returns headers for a specific request."""
        headers = self.base_headers.copy()
        # Don't send content-type for GET/HEAD if no json_data
        if method.upper() in ["GET", "HEAD"] and not has_json_data:
            headers.pop('Content-Type', None)
        return headers

    def _request(self, method: str, endpoint: str, params: Optional[Dict] = None,
                 json_data: Optional[Dict] = None, stream: bool = False,
                 allow_redirects: bool = True, timeout: int = 30) -> Union[Dict[str, Any], requests.Response, None]:
        """Makes a request to the Civitai API and handles basic errors."""
        url = f"{self.base_url}/{endpoint.lstrip('/')}"
        request_headers = self._get_request_headers(method, json_data is not None)

        try:
            response = requests.request(
                method,
                url,
                headers=request_headers,
                params=params,
                json=json_data,
                stream=stream,
                allow_redirects=allow_redirects,
                timeout=timeout
            )
            response.raise_for_status()  # Raise HTTPError for bad responses (4xx or 5xx)

            if stream:
                return response  # Return the response object for streaming

            # Handle No Content response (e.g., 204)
            if response.status_code == 204 or not response.content:
                return None

            return response.json()

        except requests.exceptions.HTTPError as http_err:
            error_detail = None
            status_code = http_err.response.status_code
            try:
                error_detail = http_err.response.json()
            except json.JSONDecodeError:
                error_detail = http_err.response.text[:200] # First 200 chars
            print(f"Civitai API HTTP Error ({method} {url}): Status {status_code}, Response: {error_detail}")
            # Return a structured error dictionary
            return {"error": f"HTTP Error: {status_code}", "details": error_detail, "status_code": status_code}

        except requests.exceptions.RequestException as req_err:
            print(f"Civitai API Request Error ({method} {url}): {req_err}")
            return {"error": str(req_err), "details": None, "status_code": None}

        except json.JSONDecodeError as json_err:
            print(f"Civitai API Error: Failed to decode JSON response from {url}: {json_err}")
            # Include response text if possible and not streaming
            response_text = response.text[:200] if not stream and hasattr(response, 'text') else "N/A"
            return {"error": "Invalid JSON response", "details": response_text, "status_code": response.status_code if hasattr(response, 'status_code') else None}

    def get_model_info(self, model_id: int) -> Optional[Dict[str, Any]]:
        """Gets information about a model by its ID. (GET /models/{id})"""
        endpoint = f"/models/{model_id}"
        result = self._request("GET", endpoint)
        # Check if the result is an error dictionary
        if isinstance(result, dict) and "error" in result:
            return result # Propagate error dict
        return result # Return model info dict or None

    def get_model_version_info(self, version_id: int) -> Optional[Dict[str, Any]]:
        """Gets information about a specific model version by its ID. (GET /model-versions/{id})"""
        endpoint = f"/model-versions/{version_id}"
        result = self._request("GET", endpoint)
        if isinstance(result, dict) and "error" in result:
            return result
        return result

    def get_model_details_trpc(self, model_id: int) -> Optional[Dict[str, Any]]:
        """Gets detailed model information, including category tags when available."""
        trpc_url = f"https://{self.domain}/api/trpc/model.getById"
        params = {
            "input": json.dumps({"json": {"id": model_id, "authed": bool(self.api_key)}})
        }
        request_headers = self._get_request_headers("GET", False)

        try:
            response = requests.get(trpc_url, params=params, headers=request_headers, timeout=30)
            response.raise_for_status()
            data = response.json()

            result_data = data.get("result", {}).get("data", {}) if isinstance(data, dict) else {}
            if isinstance(result_data, dict) and isinstance(result_data.get("json"), dict):
                return result_data["json"]

            print(f"Warning: Unexpected TRPC response structure: {data}")
            return None

        except requests.exceptions.HTTPError as http_err:
            error_detail = None
            status_code = http_err.response.status_code
            try:
                error_detail = http_err.response.json()
            except json.JSONDecodeError:
                error_detail = http_err.response.text[:200]
            print(f"Civitai TRPC HTTP Error ({trpc_url}): Status {status_code}, Response: {error_detail}")
            return {"error": f"TRPC HTTP Error: {status_code}", "details": error_detail, "status_code": status_code}

        except requests.exceptions.RequestException as req_err:
            print(f"Civitai TRPC Request Error ({trpc_url}): {req_err}")
            return {"error": str(req_err), "details": None, "status_code": None}

        except json.JSONDecodeError as json_err:
            print(f"Civitai TRPC Error: Failed to decode JSON response from {trpc_url}: {json_err}")
            response_text = response.text[:200] if hasattr(response, 'text') else "N/A"
            return {"error": "Invalid JSON response from TRPC", "details": response_text, "status_code": response.status_code if hasattr(response, 'status_code') else None}

    def extract_model_category(self, trpc_data: Dict[str, Any]) -> Optional[str]:
        """Extracts the first Civitai category tag from TRPC model data."""
        if not isinstance(trpc_data, dict):
            return None

        tags_on_models = trpc_data.get("tagsOnModels", [])
        if not isinstance(tags_on_models, list):
            return None

        for tag_entry in tags_on_models:
            tag = tag_entry.get("tag") if isinstance(tag_entry, dict) else None
            if isinstance(tag, dict) and tag.get("isCategory") is True:
                return tag.get("name")

        return None

    def search_models(self, query: str, types: Optional[List[str]] = None,
                      base_models: Optional[List[str]] = None,
                      sort: str = 'Highest Rated', period: str = 'AllTime',
                      limit: int = 20, page: int = 1,
                      nsfw: Optional[bool] = None) -> Optional[Dict[str, Any]]:
        """Searches for models on Civitai. (GET /models)"""
        endpoint = "/models"
        params = {
            "limit": max(1, min(100, limit)), # Ensure limit is reasonable
            "page": max(1, page),
            "query": query,
            "sort": sort,
            "period": period
        }
        if types:
            # `requests` handles lists by appending multiple key=value pairs,
            # which matches the expectation for 'array' type in the API doc.
             params["types"] = types
        if base_models:
            params["baseModels"] = base_models
        if nsfw is not None:
            params["nsfw"] = str(nsfw).lower() # API expects string "true" or "false"

        result = self._request("GET", endpoint, params=params)
        if isinstance(result, dict) and "error" in result:
            return result
        # Ensure structure is as expected before returning
        if isinstance(result, dict) and "items" in result and "metadata" in result:
             return result
        else:
             print(f"Warning: Unexpected search result format: {result}")
             # Return a consistent empty structure on unexpected format
             return {"items": [], "metadata": {"totalItems": 0, "currentPage": page, "pageSize": limit, "totalPages": 0}}

    def search_models_catalog(self, query: str, types: Optional[List[str]] = None,
                              base_models: Optional[List[str]] = None,
                              sort: str = 'Most Downloaded',
                              limit: int = 20, page: int = 1,
                              nsfw: Optional[bool] = None) -> Optional[Dict[str, Any]]:
        """Search using the best available endpoint for the selected domain."""
        if self.domain == "civitai.red":
            result = self.search_models(
                query=query or "",
                types=types,
                base_models=base_models,
                sort=sort,
                period="AllTime",
                limit=limit,
                page=page,
                nsfw=nsfw
            )
            if isinstance(result, dict) and "error" not in result:
                result["searchBackend"] = "rest"
            return result

        result = self.search_models_meili(
            query=query,
            types=types,
            base_models=base_models,
            sort=sort,
            limit=limit,
            page=page,
            nsfw=nsfw
        )
        if isinstance(result, dict) and "error" not in result:
            result["searchBackend"] = "meili"
        return result
        
    def search_models_meili(self, query: str, types: Optional[List[str]] = None,
                            base_models: Optional[List[str]] = None,
                            sort: str = 'metrics.downloadCount:desc', # Default to Most Downloaded
                            limit: int = 20, page: int = 1,
                            nsfw: Optional[bool] = None) -> Optional[Dict[str, Any]]:
        """Searches models using the Civitai Meilisearch endpoint."""
        meili_url = self.search_url
        headers = {'Content-Type': 'application/json'}
        headers['Authorization'] = f'Bearer {self.MEILI_TOKEN}' # Nothing harmful, everyone has the same meilisearch bearer token.

        offset = max(0, (page - 1) * limit)

        # Map simple sort terms to Meilisearch syntax
        sort_mapping = {
            "Relevancy": "id:desc",
            "Most Downloaded": "metrics.downloadCount:desc",
            "Highest Rated": "metrics.thumbsUpCount:desc", 
            "Most Liked": "metrics.favoriteCount:desc", 
            "Most Discussed": "metrics.commentCount:desc", 
            "Most Collected": "metrics.collectedCount:desc", 
            "Most Buzz": "metrics.tippedAmountCount:desc", 
            "Newest": "createdAt:desc", 
        }
        meili_sort = [sort_mapping.get(sort, "metrics.downloadCount:desc")]
        
        
        # --- Build Filters ---
        # Meilisearch uses an array of filter groups. Filters within a group are OR'd, groups are AND'd.
        filter_groups = []

        # Type Filter Group (OR logic)
        if types and isinstance(types, list) and len(types) > 0:
             # Map internal type keys/display names to API type names if needed,
             # but the provided example uses direct type names like "LORA". Let's assume frontend sends correct names.
             # Ensure proper quoting for string values in the filter.
             type_filters = [f'"type"="{t}"' for t in types]
             filter_groups.append(type_filters)

        # Base Model Filter Group (OR logic)
        if base_models and isinstance(base_models, list) and len(base_models) > 0:
            base_model_filters = [f'"version.baseModel"="{bm}"' for bm in base_models]
            filter_groups.append(base_model_filters)

        # NSFW Filter (applied as AND) - Meili typically uses boolean facets or numeric levels
        # Let's filter by 'nsfwLevel' being acceptable (1=None, 2=Mild, 4=Mature) if NSFW is false or None.
        # If NSFW is true, we don't add this filter (allow all levels).
        # This might need adjustment based on exact Meili setup and desired behavior.
        # An alternative is filtering on the 'nsfw' boolean field if it exists directly.
        if nsfw is None or nsfw is False:
             # Example: Allow levels 1, 2, 4 (adjust as needed)
             # Meili syntax for multiple values: nsfwLevel IN [1, 2, 4]
             filter_groups.append("nsfwLevel IN [1, 2, 4]") # Filter applied directly as AND
             # Or maybe filter on the boolean `nsfw` field if it's indexed:
             # filter_groups.append("nsfw = false")

        # Availability Filter (Public)
        filter_groups.append("availability = Public") # Filter applied directly as AND

        # --- Construct Request Body ---
        payload = {
            "queries": [
                {
                    "q": query if query else "", # Send empty string "" if no query
                    "indexUid": "models_v9",
                    "facets": [ # Keep facets requested by frontend if needed for analytics/refinement UI
                        "category.name",
                        "checkpointType",
                        "fileFormats",
                        "lastVersionAtUnix",
                        "tags.name",
                        "type",
                        "user.username",
                        "version.baseModel",
                        "nsfwLevel"
                    ],
                    "attributesToHighlight": [], # Keep empty if not using highlighting
                    "highlightPreTag": "__ais-highlight__",
                    "highlightPostTag": "__/ais-highlight__",
                    "limit": max(1, min(100, limit)), # Ensure reasonable limit
                    "offset": offset,
                    "filter": filter_groups
                }
            ]
        }
        if(sort != "Relevancy"):
            payload["queries"][0]["sort"] = meili_sort
        

        try:
            # print(f"DEBUG: Meili Search Payload: {json.dumps(payload, indent=2)}") # Debugging payload
            response = requests.post(meili_url, headers=headers, json=payload, timeout=25) # Use reasonable timeout
            response.raise_for_status()

            results_data = response.json()
            # print(f"DEBUG: Meili Search Response: {json.dumps(results_data, indent=2)}") # Debugging response

            # Basic validation of response structure
            if not results_data or not isinstance(results_data.get('results'), list) or not results_data['results']:
                 print(f"Warning: Meili search returned unexpected structure or empty results list: {results_data}")
                 # Return empty structure consistent with expected format downstream
                 return {"hits": [], "limit": limit, "offset": offset, "estimatedTotalHits": 0}

            # Return the content of the first result (assuming single query)
            first_result = results_data['results'][0]
            if isinstance(first_result, dict) and "hits" in first_result:
                 # Return the relevant part of the response
                 return first_result # Includes hits, processingTimeMs, limit, offset, estimatedTotalHits etc.
            else:
                  print(f"Warning: Meili search first result structure invalid: {first_result}")
                  return {"hits": [], "limit": limit, "offset": offset, "estimatedTotalHits": 0}

        except requests.exceptions.HTTPError as http_err:
            error_detail = None
            status_code = http_err.response.status_code
            try:
                error_detail = http_err.response.json()
            except json.JSONDecodeError:
                error_detail = http_err.response.text[:200]
            print(f"Civitai Meili Search HTTP Error ({meili_url}): Status {status_code}, Response: {error_detail}")
            return {"error": f"Meili HTTP Error: {status_code}", "details": error_detail, "status_code": status_code}

        except requests.exceptions.RequestException as req_err:
            print(f"Civitai Meili Search Request Error ({meili_url}): {req_err}")
            return {"error": str(req_err), "details": None, "status_code": None}

        except json.JSONDecodeError as json_err:
            print(f"Civitai Meili Search Error: Failed to decode JSON response from {meili_url}: {json_err}")
            response_text = response.text[:200] if hasattr(response, 'text') else "N/A"
            return {"error": "Invalid JSON response from Meili", "details": response_text, "status_code": response.status_code if hasattr(response, 'status_code') else None}
