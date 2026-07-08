import importlib.util
import hashlib
import sys
import tempfile
import types
import unittest
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]


def load_module(name, path):
    spec = importlib.util.spec_from_file_location(name, ROOT / path)
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


class DummyRequests(types.ModuleType):
    class exceptions:
        class HTTPError(Exception):
            pass

        class RequestException(Exception):
            pass


def install_stubs():
    requests = DummyRequests("requests")
    requests.Response = type("Response", (), {})
    requests.request = lambda *args, **kwargs: None
    requests.get = lambda *args, **kwargs: None
    requests.post = lambda *args, **kwargs: None
    sys.modules.setdefault("requests", requests)

    folder_paths = types.ModuleType("folder_paths")
    folder_paths.base_path = str(ROOT)
    folder_paths.models_dir = str(ROOT / "models")
    folder_paths.get_folder_paths = lambda model_type: []
    sys.modules.setdefault("folder_paths", folder_paths)

    aiohttp = types.ModuleType("aiohttp")
    web = types.SimpleNamespace(
        HTTPBadRequest=type("HTTPBadRequest", (Exception,), {}),
        HTTPNotFound=type("HTTPNotFound", (Exception,), {}),
    )
    aiohttp.web = web
    sys.modules.setdefault("aiohttp", aiohttp)
    sys.modules.setdefault("aiohttp.web", web)

    package = types.ModuleType("civicomfy_test")
    package.__path__ = [str(ROOT)]
    sys.modules.setdefault("civicomfy_test", package)

    config = types.ModuleType("civicomfy_test.config")
    config.PLUGIN_ROOT = str(ROOT)
    config.MODEL_TYPE_DIRS = {}
    config.DEFAULT_CHUNK_SIZE = 1024
    config.DOWNLOAD_TIMEOUT = 60
    config.HEAD_REQUEST_TIMEOUT = 25
    sys.modules.setdefault("civicomfy_test.config", config)

    for subpackage in ("api", "utils", "server", "downloader"):
        module = types.ModuleType(f"civicomfy_test.{subpackage}")
        module.__path__ = [str(ROOT / subpackage)]
        sys.modules.setdefault(f"civicomfy_test.{subpackage}", module)


install_stubs()
civitai = load_module("civicomfy_test.api.civitai", "api/civitai.py")
helpers = load_module("civicomfy_test.utils.helpers", "utils/helpers.py")
server_utils = load_module("civicomfy_test.server.utils", "server/utils.py")
chunk_downloader = load_module("civicomfy_test.downloader.chunk_downloader", "downloader/chunk_downloader.py")


class DomainSupportTests(unittest.TestCase):
    def test_normalize_domain_allows_com_and_red_only(self):
        self.assertEqual(civitai.CivitaiAPI.normalize_domain("civitai.com"), "civitai.com")
        self.assertEqual(civitai.CivitaiAPI.normalize_domain("https://civitai.red/"), "civitai.red")
        self.assertEqual(civitai.CivitaiAPI.normalize_domain("evil.example"), "civitai.com")

    def test_opposite_domain(self):
        self.assertEqual(civitai.CivitaiAPI.opposite_domain("civitai.com"), "civitai.red")
        self.assertEqual(civitai.CivitaiAPI.opposite_domain("civitai.red"), "civitai.com")

    def test_parse_civitai_red_url(self):
        model_id, version_id = helpers.parse_civitai_input("https://civitai.red/models/123?modelVersionId=456")
        self.assertEqual(model_id, 123)
        self.assertEqual(version_id, 456)

    def test_search_fallback_settings(self):
        payload = {
            "civitai_domain": "civitai.red",
            "search_opposite_on_empty": True,
            "search_opposite_on_error": True,
        }
        self.assertEqual(server_utils.resolve_civitai_domain(payload), "civitai.red")
        self.assertTrue(server_utils.resolve_search_opposite_on_empty(payload))
        self.assertTrue(server_utils.resolve_search_opposite_on_error(payload))

    def test_search_catalog_uses_rest_for_red_and_meili_for_com(self):
        class FakeAPI(civitai.CivitaiAPI):
            def __init__(self, domain):
                super().__init__(domain=domain)
                self.used = None

            def search_models(self, **kwargs):
                self.used = "rest"
                return {"items": [], "metadata": {"totalItems": 0}}

            def search_models_meili(self, **kwargs):
                self.used = "meili"
                return {"hits": [], "estimatedTotalHits": 0}

        red_api = FakeAPI("civitai.red")
        red_api.search_models_catalog(query="x")
        self.assertEqual(red_api.used, "rest")

        com_api = FakeAPI("civitai.com")
        com_api.search_models_catalog(query="x")
        self.assertEqual(com_api.used, "meili")


class CustomPathTests(unittest.TestCase):
    def test_process_custom_download_path(self):
        result = server_utils.process_custom_download_path(
            "{model_type}/{base_model}/{model_category}/{model_name}",
            {"name": "A/Model:Name"},
            {"baseModel": "SDXL 1.0"},
            "Characters",
            "lora",
        )
        self.assertEqual(result, "lora/SDXL_1.0/Characters/A_Model_Name")

    def test_process_custom_download_path_strips_traversal_and_unknown_variables(self):
        result = server_utils.process_custom_download_path("../{unknown}/{model_name}/..", {"name": "Safe"}, {}, None, "")
        self.assertEqual(result, "Safe")


class DownloadEngineTests(unittest.TestCase):
    def test_download_engine_normalization(self):
        downloader_cls = chunk_downloader.ChunkDownloader
        self.assertEqual(downloader_cls._normalize_download_engine("aria2"), "aria2")
        self.assertEqual(downloader_cls._normalize_download_engine("builtin"), "builtin")
        self.assertEqual(downloader_cls._normalize_download_engine("nope"), "auto")

    def test_final_file_validation_uses_sha256(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            file_path = Path(tmpdir) / "model.safetensors"
            payload = b"civicomfy"
            file_path.write_bytes(payload)
            expected = hashlib.sha256(payload).hexdigest()

            downloader = chunk_downloader.ChunkDownloader(
                "https://civitai.com/api/download/models/1",
                str(file_path),
                known_size=len(payload),
                expected_hashes={"SHA256": expected},
            )
            self.assertTrue(downloader._validate_final_file())

            bad_downloader = chunk_downloader.ChunkDownloader(
                "https://civitai.com/api/download/models/1",
                str(file_path),
                known_size=len(payload),
                expected_hashes={"SHA256": "0" * 64},
            )
            self.assertFalse(bad_downloader._validate_final_file())


if __name__ == "__main__":
    unittest.main()
