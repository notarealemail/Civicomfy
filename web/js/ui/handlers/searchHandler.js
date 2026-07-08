import { CivitaiDownloaderAPI } from "../../api/civitai.js";

export async function handleSearchSubmit(ui) {
    ui.searchSubmitButton.disabled = true;
    ui.searchSubmitButton.textContent = 'Searching...';
    ui.searchResultsContainer.innerHTML = '<p><i class="fas fa-spinner fa-spin"></i> Searching...</p>';
    ui.searchPaginationContainer.innerHTML = '';
    ui.ensureFontAwesome();

    const params = {
        query: ui.searchQueryInput.value.trim(),
        model_types: ui.searchTypeSelect.value === 'any' ? [] : [ui.searchTypeSelect.value],
        base_models: ui.searchBaseModelSelect.value === 'any' ? [] : [ui.searchBaseModelSelect.value],
        sort: ui.searchSortSelect.value,
        limit: ui.searchPagination.limit,
        page: ui.searchPagination.currentPage,
        api_key: ui.settings.apiKey,
        civitai_domain: ui.settings.civitaiDomain,
        search_opposite_on_empty: ui.settings.searchOppositeOnEmpty === true,
        search_opposite_on_error: ui.settings.searchOppositeOnError === true,
    };

    try {
        const response = await CivitaiDownloaderAPI.searchModels(params);
        if (!response || !response.metadata || !Array.isArray(response.items)) {
            console.error("Invalid search response structure:", response);
            throw new Error("Received invalid data from search API.");
        }

        ui.renderSearchResults(response.items);
        if (response.metadata?.fallbackFromDomain && response.metadata?.sourceDomain) {
            const notice = document.createElement('p');
            notice.className = 'civitai-search-domain-notice';
            const reason = response.metadata.fallbackReason === 'error' ? 'Search errored' : 'No results';
            notice.textContent = `${reason} on ${response.metadata.fallbackFromDomain}; showing results from ${response.metadata.sourceDomain}.`;
            ui.searchResultsContainer.prepend(notice);
        }
        ui.renderSearchPagination(response.metadata);

    } catch (error) {
        const message = `Search failed: ${error.details || error.message || 'Unknown error'}`;
        console.error("Search Submit Error:", error);
        ui.searchResultsContainer.innerHTML = `<p style="color: var(--error-text, #ff6b6b);">${message}</p>`;
        ui.showToast(message, 'error');
    } finally {
        ui.searchSubmitButton.disabled = false;
        ui.searchSubmitButton.textContent = 'Search';
    }
}
