# Convenience helper: start the whole stack and show the startup banner
# with working links in the console (docker compose up -d is detached, so
# it would otherwise print nothing).
#
# Usage:  .\up.ps1     (from the engineering directory)

docker compose up -d

# Run the banner service in the FOREGROUND: it waits for each service to
# answer and prints the links. --rm cleans the container afterwards.
docker compose run --rm banner
