# Netlify Environment Variables Setup

## Required Environment Variables

When deploying to Netlify, you can set these environment variables in your Netlify dashboard:

### Go to: Site Settings → Environment Variables

| Variable Name | Production Value | Description |
|---------------|------------------|-------------|
| `VITE_API_BASE` | *(leave empty)* | API base URL - empty uses relative `/api` |
| `VITE_APP_NAME` | `AgriAI Platform` | Application name |
| `VITE_APP_VERSION` | `1.0.0` | Application version |
| `VITE_ENABLE_MAGIC_LINK` | `true` | Enable magic link authentication |
| `VITE_ENABLE_ANALYTICS` | `true` | Enable analytics in production |
| `VITE_DEBUG_MODE` | `false` | Disable debug mode in production |

## Local Development

Copy `.env.example` to `.env` and update values:

```bash
cp .env.example .env
```

## Notes

- All environment variables starting with `VITE_` are exposed to the client
- Never put sensitive data in `VITE_` variables
- Production values are set in `netlify.toml` for automatic deployment
- Local development uses `.env` file values