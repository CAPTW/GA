from __future__ import annotations

import hashlib
import shutil
from dataclasses import dataclass
from pathlib import Path
from typing import Protocol

from .settings import OpsSettings


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


@dataclass(slots=True)
class StoredObject:
    provider: str
    object_key: str
    uri: str
    size_bytes: int
    checksum_sha256: str


class ObjectStore(Protocol):
    def ensure_ready(self) -> None: ...

    def upload_file(self, source_path: Path, object_key: str) -> StoredObject: ...


class LocalObjectStore:
    def __init__(self, root: Path) -> None:
        self._root = root

    def ensure_ready(self) -> None:
        self._root.mkdir(parents=True, exist_ok=True)

    def upload_file(self, source_path: Path, object_key: str) -> StoredObject:
        self.ensure_ready()
        destination = self._root / object_key
        destination.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(source_path, destination)
        checksum = sha256_file(destination)
        return StoredObject(
            provider="local",
            object_key=object_key,
            uri=str(destination),
            size_bytes=destination.stat().st_size,
            checksum_sha256=checksum,
        )


class S3ObjectStore:
    def __init__(
        self,
        *,
        bucket: str,
        endpoint_url: str | None,
        region_name: str | None,
        access_key_id: str | None,
        secret_access_key: str | None,
    ) -> None:
        self._bucket = bucket
        self._endpoint_url = endpoint_url
        self._region_name = region_name
        self._access_key_id = access_key_id
        self._secret_access_key = secret_access_key
        self._client = None

    def _boto_client(self):
        if self._client is None:
            try:
                import boto3
            except ImportError as exc:
                raise RuntimeError(
                    "S3 object storage requires the optional 'boto3' dependency"
                ) from exc
            self._client = boto3.client(
                "s3",
                endpoint_url=self._endpoint_url,
                region_name=self._region_name,
                aws_access_key_id=self._access_key_id,
                aws_secret_access_key=self._secret_access_key,
            )
        return self._client

    def ensure_ready(self) -> None:
        client = self._boto_client()
        try:
            client.head_bucket(Bucket=self._bucket)
        except Exception:
            create_kwargs = {"Bucket": self._bucket}
            if self._region_name and self._region_name != "us-east-1":
                create_kwargs["CreateBucketConfiguration"] = {
                    "LocationConstraint": self._region_name
                }
            client.create_bucket(**create_kwargs)

    def upload_file(self, source_path: Path, object_key: str) -> StoredObject:
        client = self._boto_client()
        self.ensure_ready()
        checksum = sha256_file(source_path)
        client.upload_file(str(source_path), self._bucket, object_key)
        return StoredObject(
            provider="s3",
            object_key=object_key,
            uri=f"s3://{self._bucket}/{object_key}",
            size_bytes=source_path.stat().st_size,
            checksum_sha256=checksum,
        )


def build_object_store(settings: OpsSettings) -> ObjectStore:
    provider = settings.object_store_provider.strip().lower()
    if provider == "local":
        return LocalObjectStore(settings.object_store_root)
    if provider == "s3":
        return S3ObjectStore(
            bucket=settings.object_store_bucket,
            endpoint_url=settings.s3_endpoint_url,
            region_name=settings.s3_region,
            access_key_id=settings.s3_access_key_id,
            secret_access_key=settings.s3_secret_access_key,
        )
    raise ValueError("GA_LAB_OBJECT_STORE_PROVIDER must be 'local' or 's3'")
