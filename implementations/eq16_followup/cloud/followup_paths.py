"""Reject ambiguous S3 paths and any overlap with frozen parent results."""
import argparse
import re
from urllib.parse import urlsplit


def parse_uri(value):
    uri = urlsplit(value)
    key = uri.path.strip("/")
    if (uri.scheme != "s3" or not re.fullmatch(r"[a-z0-9][a-z0-9.-]{1,61}[a-z0-9]", uri.netloc)
            or uri.query or uri.fragment or not key
            or not re.fullmatch(r"[A-Za-z0-9._/-]+", key)
            or any(part in ("", ".", "..") for part in key.split("/"))
            or any(char.isspace() for char in value)):
        raise ValueError("Use an S3 bucket and nonempty canonical prefix: " + value)
    return uri.netloc, key


def overlap(left, right):
    return left[0] == right[0] and (
        left[1] == right[1] or left[1].startswith(right[1] + "/")
        or right[1].startswith(left[1] + "/")
    )


def validate_paths(source, destination, logs):
    parent, output, log_output = map(parse_uri, (source, destination, logs))
    if overlap(parent, output) or overlap(parent, log_output):
        raise ValueError("Follow-up output/log prefixes must not overlap the frozen parent results")
    if overlap(output, log_output):
        raise ValueError("Follow-up result and log prefixes must be separate")
    if output[0] != log_output[0]:
        raise ValueError("Follow-up results and logs must use the same bucket")
    return parent, output, log_output


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", required=True)
    parser.add_argument("--destination", required=True)
    parser.add_argument("--logs", required=True)
    parser.add_argument("--output-bucket")
    args = parser.parse_args()
    try:
        _, destination, _ = validate_paths(args.source, args.destination, args.logs)
        if args.output_bucket and destination[0] != args.output_bucket:
            raise ValueError("Configured S3_BUCKET must match the follow-up output bucket")
    except ValueError as error:
        parser.error(str(error))


if __name__ == "__main__":
    main()
