# ================================================
# File: server/routes/SearchModels.py
# ================================================
import json
import math
import traceback
from aiohttp import web

import server # ComfyUI server instance
from ..utils import (
    get_request_json,
    resolve_civitai_api_key,
    resolve_civitai_domain,
    resolve_search_opposite_on_empty,
    resolve_search_opposite_on_error,
)
from ...api.civitai import CivitaiAPI
from ...config import CIVITAI_API_TYPE_MAP

prompt_server = server.PromptServer.instance

@prompt_server.routes.post("/civitai/search")
async def route_search_models(request):
    """API Endpoint for searching models using the selected Civitai domain."""
    try:
        data = await get_request_json(request)

        query = data.get("query", "").strip()
        model_type_keys = data.get("model_types", []) # e.g., ["lora", "checkpoint"] (frontend internal keys)
        base_model_filters = data.get("base_models", []) # e.g., ["SD 1.5", "Pony"]
        sort = data.get("sort", "Most Downloaded") # Frontend display value
        # Make period optional or remove if not supported by Meili sort directly
        # period = data.get("period", "AllTime")
        limit = int(data.get("limit", 20))
        page = int(data.get("page", 1))
        resolved_api_key = resolve_civitai_api_key(data)
        civitai_domain = resolve_civitai_domain(data)
        search_opposite_on_empty = resolve_search_opposite_on_empty(data)
        search_opposite_on_error = resolve_search_opposite_on_error(data)
        nsfw = data.get("nsfw", None) # Expect Boolean or None

        if not query and not model_type_keys and not base_model_filters:
             raise web.HTTPBadRequest(reason="Search requires a query or at least one filter (type or base model).")

        # API key priority: request payload > CIVITAI_API_KEY env var
        api = CivitaiAPI(resolved_api_key, domain=civitai_domain)

        # --- Prepare Filters for Civitai search API call ---

        # 1. Map internal type keys to Civitai API 'type' names (used in Meili filter)
        # This assumes Meili filters on the uppercase names like "LORA", "Checkpoint"
        api_types_filter = []
        if isinstance(model_type_keys, list) and model_type_keys and "any" not in model_type_keys:
            for key in model_type_keys:
                # Map key.lower() for robustness - use the existing map from config
                # CIVITAI_API_TYPE_MAP maps internal key -> Civitai API type name (e.g. 'lora' -> 'LORA')
                api_type = CIVITAI_API_TYPE_MAP.get(key.lower())
                # Ensure we handle cases where the map might return None or duplicate types
                if api_type and api_type not in api_types_filter:
                    api_types_filter.append(api_type)

        # 2. Base Model Filters (assume frontend sends exact names like "SD 1.5")
        valid_base_models = []
        if isinstance(base_model_filters, list) and base_model_filters:
             # Optional: Validate against known list?
             valid_base_models = [bm for bm in base_model_filters if isinstance(bm, str) and bm]
             # Example validation (optional):
             # valid_base_models = [bm for bm in base_model_filters if bm in AVAILABLE_MEILI_BASE_MODELS]
             # if len(valid_base_models) != len(base_model_filters):
             #     print("Warning: Some provided base model filters were invalid.")

        # --- Call the New API Method ---
        print(f"[Server Search] {api.domain}: query='{query if query else '<none>'}', types={api_types_filter or 'Any'}, baseModels={valid_base_models or 'Any'}, sort={sort}, nsfw={nsfw}, limit={limit}, page={page}")

        # Call the search method for the selected domain
        search_results = api.search_models_catalog(
             query=query or None, # Meili handles empty query if filters exist
             types=api_types_filter or None,
             base_models=valid_base_models or None,
             sort=sort, # Pass the frontend value, mapping happens in the API helper when needed
             limit=limit,
             page=page,
             nsfw=nsfw
        )

        def _result_items(result):
            if not isinstance(result, dict) or "error" in result:
                return []
            if isinstance(result.get("hits"), list):
                return result.get("hits") or []
            if isinstance(result.get("items"), list):
                return result.get("items") or []
            return []

        fallback_from_domain = None
        fallback_reason = None
        first_search_empty = isinstance(search_results, dict) and "error" not in search_results and len(_result_items(search_results)) == 0
        first_search_error = isinstance(search_results, dict) and "error" in search_results
        should_try_fallback = (
            page == 1
            and (
                (search_opposite_on_empty and first_search_empty)
                or (search_opposite_on_error and first_search_error)
            )
        )
        if should_try_fallback:
            fallback_from_domain = api.domain
            fallback_domain = CivitaiAPI.opposite_domain(api.domain)
            reason = "error" if first_search_error else "no results"
            fallback_reason = reason
            print(f"[Server Search] {reason} from {api.domain}; trying fallback domain {fallback_domain}.")
            fallback_api = CivitaiAPI(resolved_api_key, domain=fallback_domain)
            fallback_results = fallback_api.search_models_catalog(
                 query=query or None,
                 types=api_types_filter or None,
                 base_models=valid_base_models or None,
                 sort=sort,
                 limit=limit,
                 page=page,
                 nsfw=nsfw
            )
            if isinstance(fallback_results, dict) and "error" not in fallback_results and len(_result_items(fallback_results)) > 0:
                api = fallback_api
                search_results = fallback_results
            else:
                fallback_from_domain = None
                fallback_reason = None

        # Handle API error response from CivitaiAPI helper
        if search_results and isinstance(search_results, dict) and "error" in search_results:
             status_code = search_results.get("status_code", 500) or 500
             reason = f"Civitai API Search Error: {search_results.get('details', search_results.get('error', 'Unknown error'))}"
             raise web.HTTPException(reason=reason, status=status_code, body=json.dumps(search_results))

        # --- Process Search Response for Frontend ---
        if search_results and isinstance(search_results, dict):
              processed_items = []
              image_base_url = api.image_base_url # Base URL for images

              raw_items = _result_items(search_results)
              for raw_hit in raw_items:
                   hit = raw_hit.copy() if isinstance(raw_hit, dict) else raw_hit
                   if not isinstance(hit, dict): continue # Skip invalid hits

                   if search_results.get("searchBackend") == "rest":
                       creator = hit.get("creator")
                       if isinstance(creator, dict) and "user" not in hit:
                           hit["user"] = {"username": creator.get("username")}
                       if isinstance(hit.get("stats"), dict) and "metrics" not in hit:
                           hit["metrics"] = hit.get("stats")
                       versions = hit.get("modelVersions") or hit.get("versions") or []
                       if isinstance(versions, list):
                           hit["versions"] = versions
                           if versions and "version" not in hit:
                               hit["version"] = versions[0]
                           if not hit.get("images"):
                               first_version = versions[0] if versions else {}
                               if isinstance(first_version, dict) and isinstance(first_version.get("images"), list):
                                   hit["images"] = first_version.get("images")

                   thumbnail_url = None
                   # Get thumbnail from images array (prefer first image)
                   images = hit.get("images")
                   if images and isinstance(images, list) and len(images) > 0:
                       first_image = images[0]
                       # Ensure first image is a dict with a 'url' field
                       if isinstance(first_image, dict) and first_image.get("url"):
                           image_id = first_image["url"]
                           # Construct URL with a default width (e.g., 256 or 450)
                           thumbnail_url = image_id if str(image_id).startswith("http") else f"{image_base_url}/{image_id}/width=256" # Adjust width as needed

                   # Extract latest version info (Meili response includes 'version' object for the primary version)
                   latest_version_info = hit.get("version", {}) or {} # Ensure it's a dict

                   # Prepare item structure for frontend (can pass raw hit + extras, or build a specific structure)
                   # Let's pass the raw `hit` and add the `thumbnailUrl` and potentially other processed fields.
                   hit['thumbnailUrl'] = thumbnail_url # Add processed thumbnail URL directly to the hit object
                   hit['sourceDomain'] = api.domain

                   # Optional: Add more processed fields if needed, e.g., formatted stats
                   # hit['processedStats'] = { ... }

                   processed_items.append(hit)

              # --- Calculate Pagination Info ---
              result_metadata = search_results.get("metadata") if isinstance(search_results.get("metadata"), dict) else {}
              total_hits = search_results.get("estimatedTotalHits", result_metadata.get("totalItems", 0))
              current_page = page # Use the requested page number
              total_pages = math.ceil(total_hits / limit) if limit > 0 else 0

              # --- Return Structure for Frontend ---
              response_data = {
                  "items": processed_items, # The array of processed hits
                  "metadata": {
                      "totalItems": total_hits,
                      "currentPage": current_page,
                      "pageSize": limit, # The limit used for the request
                      "totalPages": total_pages,
                      # Meili provides offset, limit, processingTimeMs which could also be passed if useful
                      "meiliProcessingTimeMs": search_results.get("processingTimeMs"),
                      "meiliOffset": search_results.get("offset"),
                      "searchBackend": search_results.get("searchBackend"),
                      "sourceDomain": api.domain,
                      "fallbackFromDomain": fallback_from_domain,
                      "fallbackReason": fallback_reason,
                  }
              }
              return web.json_response(response_data)
        else:
             # Handle unexpected format from API or empty results
             print(f"[Server Search] Warning: Unexpected search result format or empty hits: {search_results}")
             return web.json_response({"items": [], "metadata": {"totalItems": 0, "currentPage": page, "pageSize": limit, "totalPages": 0}}, status=500)

    # --- Keep existing error handlers ---
    except web.HTTPError as http_err:
         # ... (keep existing HTTP error handling) ...
         body_detail = ""
         try:
              body_detail = await http_err.text() if hasattr(http_err, 'text') else http_err.body.decode('utf-8', errors='ignore') if http_err.body else ""
              if body_detail.startswith('{') and body_detail.endswith('}'): body_detail = json.loads(body_detail)
         except Exception: pass
         return web.json_response({"error": http_err.reason, "details": body_detail or "No details", "status_code": http_err.status}, status=http_err.status)

    except Exception as e:
        # ... (keep existing generic error handling) ...
        print("--- Unhandled Error in /civitai/search ---")
        traceback.print_exc()
        print("--- End Error ---")
        return web.json_response({"error": "Internal Server Error", "details": f"An unexpected search error occurred: {str(e)}", "status_code": 500}, status=500)
