"""Configuration loader with multi-level hierarchy."""

import os
from pathlib import Path
from typing import Any

import yaml
from dotenv import load_dotenv

# Load environment variables from .env file
load_dotenv()


class ConfigLoader:
    """Load and merge configuration from multiple sources.

    Hierarchy (later overrides earlier):
    1. Default config (config/default.yaml)
    2. Environment config (config/environments/{env}.yaml)
    3. Tenant config (config/tenants/{tenant_id}.yaml) [optional]
    4. Job-specific config (external source) [optional]
    5. Environment variables (RSAS_*)
    """

    def __init__(self, config_dir: Path | str | None = None):
        """Initialize config loader.

        Args:
            config_dir: Configuration directory (defaults to ./config)
        """
        if config_dir is None:
            # Default to config directory in project root
            config_dir = Path(__file__).parent.parent.parent.parent / "config"
        self.config_dir = Path(config_dir)

    def load(
        self,
        job_id: str | None = None,
        tenant_id: str | None = None,
        job_config: dict[str, Any] | None = None,
    ) -> dict[str, Any]:
        """Load configuration with full hierarchy.

        Args:
            job_id: Job identifier (for job-specific config)
            tenant_id: Tenant identifier (for tenant-specific config)
            job_config: Job-specific configuration dict (provided programmatically)

        Returns:
            Merged configuration dictionary
        """
        # 1. Load default config
        config = self._load_yaml(self.config_dir / "default.yaml")

        # 2. Merge environment-specific config
        env = os.getenv("RSAS_ENV", "development")
        env_config_path = self.config_dir / f"environments/{env}.yaml"
        if env_config_path.exists():
            env_config = self._load_yaml(env_config_path)
            config = self._deep_merge(config, env_config)

        # 3. Merge tenant config (if applicable)
        if tenant_id:
            tenant_config_path = self.config_dir / f"tenants/{tenant_id}.yaml"
            if tenant_config_path.exists():
                tenant_config = self._load_yaml(tenant_config_path)
                config = self._deep_merge(config, tenant_config)

        # 4. Merge job-specific config (if provided programmatically)
        if job_config:
            config = self._deep_merge(config, job_config)

        # 5. Override with environment variables
        config = self._apply_env_overrides(config)

        return config

    def _load_yaml(self, path: Path) -> dict[str, Any]:
        """Load YAML file.

        Args:
            path: Path to YAML file

        Returns:
            Parsed YAML as dictionary
        """
        if not path.exists():
            return {}

        with open(path) as f:
            content = yaml.safe_load(f)
            return content if content else {}

    def _deep_merge(self, base: dict[str, Any], override: dict[str, Any]) -> dict[str, Any]:
        """Deep merge two dictionaries.

        Args:
            base: Base dictionary
            override: Override dictionary

        Returns:
            Merged dictionary (base is not modified)
        """
        result = base.copy()
        for key, value in override.items():
            if key in result and isinstance(result[key], dict) and isinstance(value, dict):
                # Recursively merge nested dicts
                result[key] = self._deep_merge(result[key], value)
            else:
                # Override value
                result[key] = value
        return result

    def _apply_env_overrides(self, config: dict[str, Any]) -> dict[str, Any]:
        """Override config with environment variables.

        Environment variables starting with RSAS_ will override config values.
        Example: RSAS_OPENAI_MODEL overrides config["openai"]["model"]

        Args:
            config: Configuration dictionary

        Returns:
            Config with environment variable overrides
        """
        for key, value in os.environ.items():
            if key.startswith("RSAS_"):
                # Remove RSAS_ prefix and split by underscore
                path = key[5:].lower().split("_")
                self._set_nested(config, path, value)

        return config

    def _set_nested(
        self, config: dict[str, Any], path: list[str], value: str
    ) -> None:
        """Set nested configuration value.

        Args:
            config: Configuration dictionary
            path: Path to nested key (e.g., ["openai", "model"])
            value: Value to set
        """
        current = config
        for i, key in enumerate(path[:-1]):
            if key not in current:
                current[key] = {}
            elif not isinstance(current[key], dict):
                # Can't traverse non-dict
                return
            current = current[key]

        # Set final value, attempting type conversion
        final_key = path[-1]
        current[final_key] = self._convert_value(value)

    def _convert_value(self, value: str) -> Any:
        """Convert string value to appropriate type.

        Args:
            value: String value from environment variable

        Returns:
            Converted value (bool, int, float, or str)
        """
        # Boolean
        if value.lower() in ("true", "yes", "1"):
            return True
        if value.lower() in ("false", "no", "0"):
            return False

        # Number
        try:
            if "." in value:
                return float(value)
            return int(value)
        except ValueError:
            pass

        # String
        return value


# Global config loader instance
_config_loader: ConfigLoader | None = None


def get_config_loader() -> ConfigLoader:
    """Get global configuration loader instance.

    Returns:
        ConfigLoader: Global config loader
    """
    global _config_loader
    if _config_loader is None:
        _config_loader = ConfigLoader()
    return _config_loader


def load_config(
    job_id: str | None = None,
    tenant_id: str | None = None,
    job_config: dict[str, Any] | None = None,
) -> dict[str, Any]:
    """Load configuration (convenience function).

    Args:
        job_id: Job identifier
        tenant_id: Tenant identifier
        job_config: Job-specific configuration provided programmatically

    Returns:
        Merged configuration dictionary
    """
    loader = get_config_loader()
    return loader.load(job_id=job_id, tenant_id=tenant_id, job_config=job_config)
