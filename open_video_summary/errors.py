"""Internal failures exposed by the provider anti-corruption layer."""


class ProviderError(RuntimeError):
    retryable = False


class ConfigurationError(ProviderError):
    pass


class ProviderConfigurationError(ConfigurationError):
    """A provider-wide setting or billing failure, rather than target data."""


class AuthenticationError(ProviderError):
    pass


class RateLimitError(ProviderError):
    retryable = True


class ServiceUnavailableError(ProviderError):
    retryable = True


class ServiceTimeoutError(ProviderError):
    retryable = True


class InvalidResponseError(ProviderError):
    retryable = True


class NoSpeechError(ProviderError):
    pass
