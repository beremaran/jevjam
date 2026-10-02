# Security

Report security problems privately through
[GitHub's advisory form](https://github.com/beremaran/jevjam/security/advisories/new),
not in a public issue. Expect a first reply within a week.

Only the latest release gets fixes.

Before you report, note what jevjam already says it does:

- With `JEVJAM_API_KEY` unset, anyone who can reach the port can use the API and MCP.
  Compose binds to loopback for this reason.
- clef-flash fetches `http(s)` image and video URLs it is sent, so anyone who can
  reach the API can make it fetch from your network.
- jevjam does not serve TLS; put a reverse proxy in front for remote clients.
