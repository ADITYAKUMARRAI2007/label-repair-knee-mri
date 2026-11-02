"""Keep Google's official browser bridge alive; execute only explicit MCP calls.

Uses an in-process MCP client so this session can discover Colab tools without
changing the user's global Codex configuration. No runtime is allocated here.
"""
import asyncio
import json
import os
from pathlib import Path
import sys
import termios
from datetime import datetime, timezone

from fastmcp import Client, FastMCP
from colab_mcp.session import ColabSessionProxy

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / 'runs/colab_connection'


def emit(value):
    print(json.dumps(value, default=str), flush=True)


async def main():
    OUT.mkdir(exist_ok=True)
    if sys.stdin.isatty():
        settings = termios.tcgetattr(sys.stdin)
        settings[3] &= ~termios.ECHO
        termios.tcsetattr(sys.stdin, termios.TCSANOW, settings)
    proxy = ColabSessionProxy()
    await proxy.start_proxy_server()
    server = FastMCP(name='RSNAColabBridgeClient')
    server.mount(proxy.proxy_server)
    for middleware in proxy.middleware:
        server.add_middleware(middleware)
    try:
        async with Client(server) as client:
            async def inventory():
                tools = await client.list_tools()
                payload = [tool.model_dump(mode='json', by_alias=True) for tool in tools]
                (OUT / 'tools.json').write_text(json.dumps(payload, indent=2) + '\n')
                connected = bool(proxy.wss.connection_live.is_set())
                state = {'observed_utc':datetime.now(timezone.utc).isoformat(),
                         'pid':os.getpid(),'browser_connected':connected,
                         'tool_names':[tool.name for tool in tools]}
                (OUT / 'state.json').write_text(json.dumps(state, indent=2) + '\n')
                emit(state)
            emit({'status':'OPENING_OFFICIAL_COLAB_CONNECTION',
                  'user_action':'Approve the agent connection in the Colab browser tab.',
                  'gpu_job_started':False})
            result = await client.call_tool('open_colab_browser_connection', {})
            emit({'connection_result':result.data})
            await inventory()
            emit({'status':'BRIDGE_READY_FOR_EXPLICIT_COMMANDS'})
            while True:
                line = await asyncio.to_thread(sys.stdin.readline)
                if not line:
                    break
                try:
                    request = json.loads(line)
                    if request['action'] == 'stop':
                        break
                    if request['action'] == 'list':
                        await inventory()
                    elif request['action'] == 'call':
                        result = await client.call_tool(request['name'], request.get('arguments', {}))
                        emit({'result':result.model_dump(mode='json') if hasattr(result,'model_dump') else str(result)})
                    else:
                        raise ValueError('Unknown action')
                except Exception as error:
                    emit({'error_type':type(error).__name__,'message':str(error)})
    finally:
        await proxy.cleanup()
        emit({'status':'BRIDGE_STOPPED'})


if __name__ == '__main__':
    asyncio.run(main())
