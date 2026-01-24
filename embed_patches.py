# /// script
# requires-python = ">=3.12"
# ///
"""Embed an Astral-specific SBOM into a wheel's .dist-info/sboms/ directory."""

import argparse
import base64
import csv
import email
import hashlib
import io
import json
import os
import zipfile
from pathlib import Path


def rewrite_zip_with_bytes(
    src_zip_path: str | os.PathLike[str],
    dst_zip_path: str | os.PathLike[str],
    target_filenames: dict[str, str],
    additions: dict[str, str] | None = None,
) -> None:
    """Rewrite a zip file, replacing and adding files.

    Args:
        src_zip_path: Path to the source zip file.
        dst_zip_path: Path to the destination zip file.
        target_filenames: Map of filename -> content for files to replace.
        additions: Map of filename -> content for files to add.
    """
    with (
        zipfile.ZipFile(src_zip_path, "r") as zin,
        zipfile.ZipFile(dst_zip_path, "w") as zout,
    ):
        for original_info in zin.infolist():
            compress_type = original_info.compress_type

            if data := target_filenames.get(original_info.filename):
                # Write out the updated data.
                zout.writestr(
                    original_info,
                    data,
                    compress_type=compress_type,
                )
            else:
                # Read the existing data.
                data = zin.read(original_info.filename)

                # Copy over the existing data.
                zout.writestr(original_info, data, compress_type=compress_type)

        # Add new files.
        if additions:
            for filename, content in additions.items():
                zout.writestr(filename, content, compress_type=zipfile.ZIP_DEFLATED)


def get_dist_info_dir(wheel: zipfile.ZipFile) -> str:
    """Find the .dist-info directory in a wheel."""
    for name in wheel.namelist():
        if name.endswith(".dist-info/METADATA"):
            return name.rsplit("/", 1)[0]
    raise ValueError("Could not find .dist-info directory in wheel")


def compute_hash(content: bytes) -> str:
    """Compute the hash for a RECORD entry."""
    digest = hashlib.sha256(content).digest()
    return f"sha256={base64.urlsafe_b64encode(digest).rstrip(b'=').decode()}"


def augment_metadata(
    metadata_content: str,
    source_repo: str,
    source_commit: str,
    patches: dict[str, str],
) -> str:
    """Augment the wheel metadata with provenance information.

    Args:
        metadata_content: The original METADATA content.
        source_repo: The source repository URL.
        source_commit: The source commit SHA.
        patches: Map of patch filename -> patch content.

    Returns:
        The augmented METADATA content.
    """
    # Parse the metadata.
    wheel_metadata = email.message_from_string(metadata_content)

    # Store the existing description.
    description = wheel_metadata.get_payload() or ""
    assert isinstance(description, str)
    wheel_metadata.set_payload("")

    # Set the description content-type to Markdown.
    match wheel_metadata.get("Description-Content-Type"):
        case None:
            wheel_metadata.add_header("Description-Content-Type", "text/markdown")
        case "text/markdown":
            pass
        case value:
            raise ValueError(f"Unexpected `Description-Content-Type` header: {value}")

    # Add provenance to the description.
    if description:
        description += "\n\n"
        description += "---"
        description += "\n\n"

    # Extract `owner/repo` from the repository URL (e.g.,
    # `https://github.com/open-mmlab/mmcv` -> `open-mmlab/mmcv`).
    repo_name = "/".join(source_repo.rstrip("/").split("/")[-2:])
    truncated_sha = source_commit[:7]
    commit_url = f"{source_repo.rstrip('/')}/commit/{source_commit}"

    description += (
        f"This distribution was built by Astral from [{repo_name}@{truncated_sha}]({commit_url})."
    )

    if patches:
        description += "\n\n"
        description += "The following patches were applied to the upstream source:\n"
        for patch_name, patch_content in sorted(patches.items()):
            description += f"\n**{patch_name}**\n"
            description += f"```diff\n{patch_content}```\n"

    description += "\n"
    wheel_metadata.set_payload(description)

    return wheel_metadata.as_string()


def embed_sbom(
    wheel_path: Path,
    patches_dir: Path | None,
    source_repo: str,
    source_tag: str,
    source_commit: str,
    build_repo: str,
    build_commit: str,
) -> None:
    """Embed an SBOM into a wheel file."""
    # Find all .patch files.
    patch_files = sorted(patches_dir.glob("*.patch")) if patches_dir else []

    # Build the patches dict for both SBOM and description augmentation.
    patches = {patch_file.name: patch_file.read_text() for patch_file in patch_files}

    # Build the SBOM.
    sbom = {
        "source": {
            "repository": source_repo,
            "tag": source_tag,
            "commit": source_commit,
        },
        "build": {
            "repository": build_repo,
            "commit": build_commit,
        },
        "patches": patches,
    }
    sbom_content = json.dumps(sbom, indent=2) + "\n"
    sbom_bytes = sbom_content.encode("utf-8")

    # Determine paths within the wheel.
    with zipfile.ZipFile(wheel_path, "r") as wheel:
        dist_info = get_dist_info_dir(wheel)
        metadata_path = f"{dist_info}/METADATA"
        record_path = f"{dist_info}/RECORD"
        sbom_path = f"{dist_info}/sboms/astral.json"

        # Read and augment METADATA.
        metadata_content = wheel.read(metadata_path).decode("utf-8")
        new_metadata = augment_metadata(
            metadata_content,
            source_repo=source_repo,
            source_commit=source_commit,
            patches=patches,
        )
        new_metadata_bytes = new_metadata.encode("utf-8")
        metadata_hash = compute_hash(new_metadata_bytes)

        # Read existing RECORD.
        record_content = wheel.read(record_path).decode("utf-8")
        record_lines = record_content.splitlines()

        # Build new RECORD entries.
        record_out = io.StringIO()
        reader = csv.reader(record_lines)
        writer = csv.writer(record_out)
        for row in reader:
            if not row:
                continue
            path = row[0]
            # Skip RECORD itself (added at end) and METADATA (will be updated).
            if path == record_path:
                continue
            if path == metadata_path:
                # Update METADATA entry with new hash and size.
                writer.writerow([metadata_path, metadata_hash, str(len(new_metadata_bytes))])
            else:
                writer.writerow(row)

        # Add SBOM entry.
        sbom_hash = compute_hash(sbom_bytes)
        writer.writerow([sbom_path, sbom_hash, str(len(sbom_bytes))])

        writer.writerow([record_path, "", ""])  # RECORD has no hash.
        new_record = record_out.getvalue()

    # Rewrite the wheel.
    temp_path = wheel_path.with_suffix(".tmp")
    rewrite_zip_with_bytes(
        wheel_path,
        temp_path,
        target_filenames={
            metadata_path: new_metadata,
            record_path: new_record,
        },
        additions={sbom_path: sbom_content},
    )

    # Replace original wheel.
    temp_path.replace(wheel_path)
    print(f"Embedded SBOM with {len(patch_files)} patch(es) into {wheel_path}")


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Embed an Astral-specific SBOM into a wheel's .dist-info/sboms/ directory."
    )
    parser.add_argument("wheel", type=Path, help="Path to the wheel file")
    parser.add_argument("patches_dir", type=Path, help="Path to the patches directory")
    parser.add_argument(
        "--source-repo", type=str, required=True, help="Source repository URL"
    )
    parser.add_argument(
        "--source-tag", type=str, required=True, help="Source git tag"
    )
    parser.add_argument(
        "--source-commit", type=str, required=True, help="Source git commit SHA"
    )
    parser.add_argument(
        "--build-repo", type=str, required=True, help="Build repository URL"
    )
    parser.add_argument(
        "--build-commit", type=str, required=True, help="Build git commit SHA"
    )
    args = parser.parse_args()

    if not args.wheel.exists():
        parser.error(f"Wheel not found: {args.wheel}")

    # If patches directory doesn't exist, use None to indicate no patches.
    patches_dir = args.patches_dir if args.patches_dir.exists() else None

    embed_sbom(
        args.wheel,
        patches_dir,
        source_repo=args.source_repo,
        source_tag=args.source_tag,
        source_commit=args.source_commit,
        build_repo=args.build_repo,
        build_commit=args.build_commit,
    )


if __name__ == "__main__":
    main()
