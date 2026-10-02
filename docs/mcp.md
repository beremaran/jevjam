# MCP server

Back to the [README](../README.md).

The same models serve [MCP](https://modelcontextprotocol.io) tools at `/mcp` over
streamable HTTP. Laya's own `laya-mcp-server` speaks stdio; this app reuses its tool
handlers but serves them over HTTP so the server can keep running independently of a
client. The server calls itself `jevjam`, and four tools are available. They were
named `laya_*` before the rename; the old names are gone.

| Tool | What it does |
| --- | --- |
| `jevjam_predict` | Answer typed questions (`choice` / `score` / `noul`) over a state in one forward pass. Like `POST /v1/systemone`, but for a calling agent. `model: "julia-1"` asks Julia-1. |
| `jevjam_preset` | Run a built-in workflow on Laya: `guard`, `moderation`, `triage`, or `model_router`. |
| `jevjam_route` | Which Laya checkpoint would answer, without loading one. |
| `jevjam_status` | Device in use, package versions, and what is resident. |

Those are the calls that run a forward pass (`jevjam_predict` directly, `jevjam_preset`
through it), so they are the ones that keep a checkpoint resident and reset the idle
clock.

## Authentication and remote access

When `JEVJAM_API_KEY` is set, every request to `/mcp` must include
`Authorization: Bearer <key>`. The same key protects the HTTP API. Leave it unset to
allow unauthenticated requests; `GET /health` stays public either way.

For local clients, use `http://127.0.0.1:8000/mcp`. For remote clients, point them at
the proxy URL. The proxy handles TLS and routes to jevjam; jevjam checks the Bearer token.
The four combinations are:

| Proxy URL | `JEVJAM_API_KEY` on server | Client sends Bearer token | Result |
| --- | --- | --- | --- |
| `http://jevjam.example.com/mcp` | set | yes | Authenticated, but HTTP does not encrypt the token or traffic. Use only on a trusted private network. |
| `https://jevjam.example.com/mcp` | set | yes | Authenticated and encrypted in transit. |
| `http://jevjam.example.com/mcp` | unset | no | Open to anyone who can reach it; traffic is unencrypted. |
| `https://jevjam.example.com/mcp` | unset | no | Open to anyone who can reach it; traffic is encrypted. |

If the server key is set but the client omits it or sends a wrong one, MCP returns
`401`. Keep the key out of shared project config. Set it in the server environment
and in the environment used to start the client. For example, in a local shell:

```bash
export JEVJAM_API_KEY=secret
docker compose up -d
```

## Install in a client

The examples below use user-local settings so credentials do not go into a shared
project file. In each client, use the local URL and omit the `Authorization` header
when the server key is unset. For a remote connection, replace the URL; add the
Bearer setting when the server key is set. A client using a remote URL must be able
to reach that proxy from its own host or network.

### [OpenCode](https://opencode.ai/v2/docs/mcp-servers)

Add this to `~/.config/opencode/opencode.json`:

```json
{
  "mcp": {
    "servers": {
      "jevjam": {
        "type": "remote",
        "url": "http://127.0.0.1:8000/mcp"
      }
    }
  }
}
```

For a remote server with a key, use this entry instead. OpenCode reads the variable
from its process environment:

```json
{
  "mcp": {
    "servers": {
      "jevjam": {
        "type": "remote",
        "url": "https://jevjam.example.com/mcp",
        "oauth": false,
        "headers": {
          "Authorization": "Bearer {env:JEVJAM_API_KEY}"
        }
      }
    }
  }
}
```

Check the connection with `opencode mcp list`. You can also add a server with
`opencode mcp add jevjam --global --url http://127.0.0.1:8000/mcp`. OpenCode remote
servers use OAuth by default, so the authenticated config sets `"oauth": false` and
sends the Bearer header instead. For a project-only entry, use the same shape in the
project's `opencode.json` and omit `--global` from the CLI command.

### [Pi](https://pi.dev/)

Pi core does not include MCP support. Install the third-party
[`pi-mcp-adapter`](https://pi.dev/packages/pi-mcp-adapter) extension, then restart
Pi:

```bash
pi install npm:pi-mcp-adapter
```

Add this to the user config at `~/.config/mcp/mcp.json`:

```json
{
  "mcpServers": {
    "jevjam": {
      "url": "http://127.0.0.1:8000/mcp"
    }
  }
}
```

For a project-shared server, the adapter also reads `.mcp.json` in the project root.

For a remote server with a key, use this config instead. The adapter expands the
environment variable:

```json
{
  "mcpServers": {
    "jevjam": {
      "url": "https://jevjam.example.com/mcp",
      "headers": {
        "Authorization": "Bearer ${JEVJAM_API_KEY}"
      }
    }
  }
}
```

### [Codex](https://developers.openai.com/codex/mcp)

Add the local server to `~/.codex/config.toml`:

```bash
codex mcp add jevjam --url http://127.0.0.1:8000/mcp
```

For a remote server with a key, use the proxy URL and name the environment variable
that holds the token:

```bash
codex mcp add jevjam --url https://jevjam.example.com/mcp \
  --bearer-token-env-var JEVJAM_API_KEY
```

Check with `codex mcp list` or `/mcp` in the Codex TUI. Codex also reads project
settings from `.codex/config.toml` in trusted projects.

### [Claude Code](https://code.claude.com/docs/en/mcp)

Add the local server for all your projects:

```bash
claude mcp add --transport http --scope user jevjam http://127.0.0.1:8000/mcp
```

For a remote server with a key, use the proxy URL and pass the token from the
environment:

```bash
claude mcp add --transport http --scope user jevjam \
  https://jevjam.example.com/mcp \
  --header "Authorization: Bearer $JEVJAM_API_KEY"
```

Check with `claude mcp list` or `/mcp`. Omit `--scope user` to keep the server
private to the current project; `--scope project` writes a shared `.mcp.json`.

## Reverse proxy examples

These snippets route `/mcp` to jevjam over HTTP and pass the `Authorization` header
through. Configure TLS on the proxy as you normally would; these examples do not
set up certificates or add a second auth layer. If the proxy runs in Docker, use an
upstream address it can reach, such as `http://jevjam:8000` when both containers share
a network, instead of `127.0.0.1`.

Nginx, inside the proxy's existing `server` block:

```nginx
location /mcp {
    proxy_pass http://127.0.0.1:8000;
    proxy_http_version 1.1;
    proxy_read_timeout 10m;
    proxy_set_header Host $host;
    proxy_set_header Authorization $http_authorization;
}
```

Traefik dynamic configuration:

```yaml
http:
  routers:
    jevjam-mcp:
      rule: "Host(`jevjam.example.com`) && PathPrefix(`/mcp`)"
      service: jevjam-mcp
  services:
    jevjam-mcp:
      loadBalancer:
        servers:
          - url: "http://127.0.0.1:8000"
```

Traefik forwards `Authorization` by default. Do not add middleware that removes it.
