import base64
import json

class Base64Encoder:
    @staticmethod
    def encode(value: str) -> str:
        """Encodes a string value using Base64 encoding."""
        value_bytes = value.encode('utf-8')
        encoded_bytes = base64.b64encode(value_bytes)
        return encoded_bytes.decode('utf-8')

    @staticmethod
    def encode_values(**kwargs) -> dict:
        """Encodes multiple values using Base64 encoding."""
        encoded_values = {}
        for key, value in kwargs.items():
            if value is not None:
                encoded_values[key] = Base64Encoder.encode(value)
        return encoded_values

    @staticmethod
    def encode_values_to_json(**kwargs) -> str:
        """Encodes values using Base64 and returns JSON output."""
        encoded_values = Base64Encoder.encode_values(**kwargs)
        return json.dumps(encoded_values)
    

# Example usage:
access_key_id = "your_access_key_id"
bucket = "your_bucket_name"
path_prefix = "/path/prefix"
host = "optional_host"
region = "optional_region"
secret_access_key = "your_secret_access_key"
uri = "optional_uri"
username = "optional_username"

# Get encoded values as JSON string
encoded_json = Base64Encoder.encode_values_to_json(
    access_key_id=access_key_id,
    bucket=bucket,
    path_prefix=path_prefix,
    host=host,
    region=region,
    secret_access_key=secret_access_key,
    uri=uri,
    username=username
)
print("Encoded Values (JSON):")
print(encoded_json)