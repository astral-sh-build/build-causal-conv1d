# /// script
# requires-python = ">=3.12"
# ///
"""Embed Astral-specific metadata into a wheel's .dist-info directory."""

import argparse
import base64
import csv
import email
import hashlib
import io
import json
import os
import zipfile
from email.message import Message
from pathlib import Path, PurePosixPath


def rewrite_zip_with_bytes(
    src_zip_path: str | os.PathLike[str],
    dst_zip_path: str | os.PathLike[str],
    target_filenames: dict[str, bytes],
    additions: dict[str, bytes] | None = None,
) -> None:
    """Rewrite a zip file, replacing and adding files.

    Args:
        src_zip_path: Path to the source zip file.
        dst_zip_path: Path to the destination zip file.
        target_filenames: Map of filename -> byte content for files to replace.
        additions: Map of filename -> byte content for files to add.
    """
    with (
        zipfile.ZipFile(src_zip_path, "r") as zin,
        zipfile.ZipFile(dst_zip_path, "w") as zout,
    ):
        for original_info in zin.infolist():
            compress_type = original_info.compress_type

            if original_info.filename in target_filenames:
                # Write out the updated data.
                data = target_filenames[original_info.filename]
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


def parse_metadata_version(value: str) -> tuple[int, ...]:
    """Parse a Core Metadata version for comparison."""
    try:
        return tuple(int(part) for part in value.split("."))
    except ValueError as exc:
        raise ValueError(f"Unexpected Metadata-Version: {value}") from exc


def set_license_expression(
    wheel_metadata: Message, license_expression: str | None
) -> None:
    """Set PEP 639 license metadata."""
    if license_expression is None:
        return

    metadata_version = wheel_metadata.get("Metadata-Version")
    if metadata_version is None:
        wheel_metadata.add_header("Metadata-Version", "2.4")
    elif parse_metadata_version(metadata_version) < (2, 4):
        wheel_metadata.replace_header("Metadata-Version", "2.4")

    # Core Metadata 2.4 makes License and License-Expression mutually exclusive.
    del wheel_metadata["License"]

    if wheel_metadata.get("License-Expression") is None:
        wheel_metadata.add_header("License-Expression", license_expression)
    else:
        wheel_metadata.replace_header("License-Expression", license_expression)


def augment_metadata(
    metadata_content: str,
    source_repo: str,
    source_commit: str,
    patches: dict[str, str],
    license_expression: str | None,
    license_files: list[str],
) -> str:
    """Augment the wheel metadata with provenance information.

    Args:
        metadata_content: The original METADATA content.
        source_repo: The source repository URL.
        source_commit: The source commit SHA.
        patches: Map of patch filename -> patch content.
        license_expression: SPDX license expression to add to the metadata.
        license_files: License-File paths to add to the metadata.

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

    set_license_expression(wheel_metadata, license_expression)

    # Record license files that are embedded under .dist-info/licenses/.
    existing_license_files = set(wheel_metadata.get_all("License-File", []))
    for license_file in license_files:
        if license_file not in existing_license_files:
            wheel_metadata.add_header("License-File", license_file)
            existing_license_files.add(license_file)

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
    license_files: list[tuple[Path, str]],
    license_expression: str | None,
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
        existing_names = set(wheel.namelist())
        license_file_contents = {
            f"{dist_info}/licenses/{license_path}": source_path.read_bytes()
            for source_path, license_path in license_files
        }

        # Read and augment METADATA.
        metadata_content = wheel.read(metadata_path).decode("utf-8")
        new_metadata = augment_metadata(
            metadata_content,
            source_repo=source_repo,
            source_commit=source_commit,
            patches=patches,
            license_expression=license_expression,
            license_files=[license_path for _, license_path in license_files],
        )
        new_metadata_bytes = new_metadata.encode("utf-8")
        generated_files = {
            metadata_path: new_metadata_bytes,
            sbom_path: sbom_bytes,
            **license_file_contents,
        }

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
            # Skip RECORD itself and generated files; all are added below.
            if path == record_path:
                continue
            if path in generated_files:
                continue
            writer.writerow(row)

        for path, content in generated_files.items():
            writer.writerow([path, compute_hash(content), str(len(content))])

        writer.writerow([record_path, "", ""])  # RECORD has no hash.
        new_record = record_out.getvalue()
        new_record_bytes = new_record.encode("utf-8")

        replacements = {
            metadata_path: new_metadata_bytes,
            record_path: new_record_bytes,
        }
        additions = {}
        for path, content in {sbom_path: sbom_bytes, **license_file_contents}.items():
            if path in existing_names:
                replacements[path] = content
            else:
                additions[path] = content

    # Rewrite the wheel.
    temp_path = wheel_path.with_suffix(".tmp")
    rewrite_zip_with_bytes(
        wheel_path,
        temp_path,
        target_filenames=replacements,
        additions=additions,
    )

    # Replace original wheel.
    temp_path.replace(wheel_path)
    print(f"Embedded SBOM with {len(patch_files)} patch(es) into {wheel_path}")


def parse_license_file_spec(parser: argparse.ArgumentParser, spec: str) -> tuple[Path, str]:
    """Parse a SOURCE:DEST license-file argument."""
    if ":" not in spec:
        parser.error(f"Invalid --license-file value, expected SOURCE:DEST: {spec}")
    source, destination = spec.split(":", 1)
    source_path = Path(source)
    if not source_path.is_file():
        parser.error(f"License file not found: {source_path}")

    destination_path = PurePosixPath(destination)
    if (
        not destination
        or destination_path.is_absolute()
        or any(part in {"", ".."} for part in destination_path.parts)
    ):
        parser.error(
            f"License file destination must be a relative path under .dist-info/licenses/: {destination}"
        )

    return source_path, destination_path.as_posix()


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Embed Astral-specific metadata into a wheel's .dist-info directory."
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
    parser.add_argument(
        "--license-file",
        action="append",
        default=[],
        metavar="SOURCE:DEST",
        help=(
            "Embed SOURCE into the wheel's .dist-info/licenses/DEST and add "
            "License-File: DEST metadata. May be passed multiple times."
        ),
    )
    parser.add_argument(
        "--license-expression",
        type=str,
        help="Set the wheel's PEP 639 License-Expression metadata.",
    )
    args = parser.parse_args()

    if not args.wheel.exists():
        parser.error(f"Wheel not found: {args.wheel}")

    # If patches directory doesn't exist, use None to indicate no patches.
    patches_dir = args.patches_dir if args.patches_dir.exists() else None
    license_files = [
        parse_license_file_spec(parser, license_file)
        for license_file in args.license_file
    ]
    license_expression = (
        args.license_expression.strip()
        if args.license_expression is not None
        else None
    )
    if args.license_expression is not None and not license_expression:
        parser.error("--license-expression must not be empty")

    embed_sbom(
        args.wheel,
        patches_dir,
        license_files,
        license_expression,
        source_repo=args.source_repo,
        source_tag=args.source_tag,
        source_commit=args.source_commit,
        build_repo=args.build_repo,
        build_commit=args.build_commit,
    )


if __name__ == "__main__":
    main()
