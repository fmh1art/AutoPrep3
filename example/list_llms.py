#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
List available LLMs for an Azure/OpenAI-style endpoint.

Queries both `deployments` (Azure style) and `models` (OpenAI style) endpoints
using only Python standard library (urllib) to avoid extra dependencies.
"""

import argparse
import json
import sys

# Py2/Py3 compatibility for urllib
try:
    import urllib.request as urllib_request
    import urllib.parse as urllib_parse
    import urllib.error as urllib_error
except Exception:
    import urllib2 as urllib_request
    import urllib as urllib_parse
    import urllib2 as urllib_error


def http_get_json(url, params, headers, timeout=30):
    try:
        q = urllib_parse.urlencode(params) if params else ""
        full_url = (url + "?" + q) if q else url
        req = urllib_request.Request(full_url, headers=headers)
        # Python2 doesn't support context manager for urlopen
        resp = urllib_request.urlopen(req, timeout=timeout)
        # Python2 response object doesn't have .status; use getcode()
        status = getattr(resp, "status", None) or getattr(resp, "getcode", lambda: 200)()
        body = resp.read()
        try:
            data = json.loads(body.decode("utf-8"))
        except Exception:
            try:
                data = json.loads(body)
            except Exception:
                # Return raw text if JSON parsing fails
                try:
                    data = {"_raw": body.decode("utf-8")}
                except Exception:
                    data = {"_raw": str(body)}
        return status, data
    except urllib_error.HTTPError as e:
        try:
            err_body = e.read()
            try:
                err_body = err_body.decode("utf-8")
            except Exception:
                err_body = str(err_body)
        except Exception:
            err_body = str(e)
        code = getattr(e, "code", 0)
        return code, {"error": err_body}
    except Exception as e:
        return 0, {"error": str(e)}


def print_deployments(base_url, api_version, headers):
    url = base_url.rstrip("/") + "/deployments"
    status, data = http_get_json(url, {"api-version": api_version}, headers)
    print("[{0}] GET {1}?api-version={2}".format(status, url, api_version))
    if status != 200:
        msg = data.get("error") or data.get("_raw") if isinstance(data, dict) else None
        if msg:
            print("deployments error: {0}".format(msg[:300]))
        return False

    # Azure-style: { value: [ ... ] } or OpenAI-style: { data: [ ... ] }
    items = []
    if isinstance(data, dict):
        items = data.get("value") or data.get("data") or []
    elif isinstance(data, list):
        items = data
    if not items:
        print("deployments: empty")
        return False
    print("可用部署 (deployments):")
    for d in items:
        name = (d.get("name") or d.get("id")) if isinstance(d, dict) else str(d)
        model = None
        if isinstance(d, dict):
            model = d.get("model") or (d.get("properties", {}).get("model") if isinstance(d.get("properties", {}), dict) else None)
        suffix = " (model={0})".format(model) if model else ""
        print("- {0}{1}".format(name, suffix))
    return True


def print_models(base_url, api_version, headers):
    url = base_url.rstrip("/") + "/models"
    status, data = http_get_json(url, {"api-version": api_version}, headers)
    print("[{0}] GET {1}?api-version={2}".format(status, url, api_version))
    if status != 200:
        msg = data.get("error") or data.get("_raw") if isinstance(data, dict) else None
        if msg:
            print("models error: {0}".format(msg[:300]))
        return False

    # OpenAI-style: { data: [ ... ] } or plain list
    items = []
    if isinstance(data, dict):
        items = data.get("data") or data.get("models") or []
    elif isinstance(data, list):
        items = data
    if not items:
        print("models: empty")
        return False
    print("可用模型 (models):")
    for m in items:
        mid = (m.get("id") or m.get("name")) if isinstance(m, dict) else str(m)
        print("- {0}".format(mid))
    return True


def main():
    p = argparse.ArgumentParser(description="List available LLMs/deployments")
    p.add_argument("--base-url", required=True, help="Service base URL")
    p.add_argument("--api-version", required=True, help="API version")
    p.add_argument("--api-key", required=True, help="API key")
    p.add_argument(
        "--header",
        action="append",
        default=[],
        help="Extra header in KEY=VALUE format; can be repeated",
    )
    args = p.parse_args()

    headers = {"api-key": args.api_key}
    for h in args.header:
        if "=" in h:
            k, v = h.split("=", 1)
            headers[k.strip()] = v.strip()

    ok = print_deployments(args.base_url, args.api_version, headers)
    if not ok:
        print("deployments 未返回有效结果，尝试 models ...")
        ok2 = print_models(args.base_url, args.api_version, headers)
        if not ok2:
            print("deployments 与 models 均未返回有效数据。")


if __name__ == "__main__":
    sys.exit(main())
