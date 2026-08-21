# The worker, and nothing else.
#
# One image serves every agent in a project: which ones is a matter of the
# worker.yaml you mount, not of what was baked in. That keeps the image the same
# artifact across environments, so a config change is a restart rather than a
# rebuild.
FROM python:3.12-slim

# Agents declare MCP servers as commands, and the two common ones are `python`
# (already here) and `npx`. Node is a real cost in image size and a real
# convenience — an agent that declares an npm-published MCP server just works
# instead of failing at boot with a quarantine nobody expected.
RUN apt-get update \
 && apt-get install -y --no-install-recommends nodejs npm \
 && rm -rf /var/lib/apt/lists/*

WORKDIR /app

# Dependencies first, so editing an agent's YAML doesn't reinstall the world.
COPY pyproject.toml README.md ./
COPY charter ./charter
RUN pip install --no-cache-dir .

# The project is mounted, not copied — see the compose file. An operator changing
# an objective should not be building an image to do it.
WORKDIR /project

# `worker .` reads worker.yaml from here, and spawns stdio MCP servers relative to
# it, which is why the working directory matters rather than being incidental.
ENTRYPOINT ["charter", "worker", "."]
