import ast
import datetime
import json
import logging
import os
import platform
import subprocess
import sys
import threading
import urllib.request
import webbrowser
from config import WEATHER_DEFAULT_CITY, SEARCH_MAX_RESULTS, PYTHON_SANDBOX_TIMEOUT, PYTHON_SANDBOX_MAX_OUTPUT

logger = logging.getLogger(__name__)

TOOLS = [
    {
        "type": "function",
        "function": {
            "name": "get_time",
            "description": "Получить текущую дату и время",
            "parameters": {
                "type": "object",
                "properties": {},
                "required": [],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "get_weather",
            "description": "Узнать погоду в указанном городе",
            "parameters": {
                "type": "object",
                "properties": {
                    "city": {
                        "type": "string",
                        "description": "Название города на русском или английском",
                    },
                },
                "required": ["city"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "set_timer",
            "description": "Установить таймер на указанное количество минут",
            "parameters": {
                "type": "object",
                "properties": {
                    "minutes": {
                        "type": "integer",
                        "description": "Длительность таймера в минутах",
                    },
                    "label": {
                        "type": "string",
                        "description": "Название таймера (для напоминания)",
                    },
                },
                "required": ["minutes"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "calculate",
            "description": "Вычислить математическое выражение",
            "parameters": {
                "type": "object",
                "properties": {
                    "expression": {
                        "type": "string",
                        "description": "Математическое выражение, например '2 + 2 * 3'",
                    },
                },
                "required": ["expression"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "search_web",
            "description": "Поиск информации в интернете",
            "parameters": {
                "type": "object",
                "properties": {
                    "query": {
                        "type": "string",
                        "description": "Поисковый запрос",
                    },
                },
                "required": ["query"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "system_info",
            "description": "Получить информацию о системе (ОС, процессор, память)",
            "parameters": {
                "type": "object",
                "properties": {},
                "required": [],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "open_app",
            "description": "Открыть приложение на компьютере",
            "parameters": {
                "type": "object",
                "properties": {
                    "app": {
                        "type": "string",
                        "description": "Название приложения (например: блокнот, калькулятор, браузер, проводник)",
                    },
                },
                "required": ["app"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "run_python",
            "description": "Выполнить Python-код и вернуть результат (stdout или значение последнего выражения). Используй для вычислений, конвертации единиц, обработки данных, генерации чисел и т.д.",
            "parameters": {
                "type": "object",
                "properties": {
                    "code": {
                        "type": "string",
                        "description": "Python-код для выполнения. Если нужно вывести результат — используй print(). Иначе вернётся значение последнего выражения.",
                    },
                },
                "required": ["code"],
            },
        },
    },
]


def execute_tool(name: str, arguments: dict) -> str:
    """Выполнить инструмент по имени. Возвращает результат как строку."""
    dispatch = {
        "get_time": _get_time,
        "get_weather": _get_weather,
        "set_timer": _set_timer,
        "calculate": _calculate,
        "search_web": _search_web,
        "system_info": _system_info,
        "open_app": _open_app,
        "run_python": _run_python,
    }

    handler = dispatch.get(name)
    if handler is None:
        logger.warning("Неизвестный инструмент: %s", name)
        return f"Неизвестный инструмент: {name}"

    try:
        result = handler(**arguments)
        logger.info("Tool %s → %s", name, result[:100])
        return result
    except Exception as e:
        logger.error("Ошибка выполнения %s: %s", name, e, exc_info=True)
        return f"Ошибка: {e}"


def _get_time() -> str:
    now = datetime.datetime.now()
    return now.strftime("%A, %d %B %Y, %H:%M").replace(
        "Monday", "Понедельник"
    ).replace(
        "Tuesday", "Вторник"
    ).replace(
        "Wednesday", "Среда"
    ).replace(
        "Thursday", "Четверг"
    ).replace(
        "Friday", "Пятница"
    ).replace(
        "Saturday", "Суббота"
    ).replace(
        "Sunday", "Воскресенье"
    ).replace(
        "January", "января"
    ).replace(
        "February", "февраля"
    ).replace(
        "March", "марта"
    ).replace(
        "April", "апреля"
    ).replace(
        "May", "мая"
    ).replace(
        "June", "июня"
    ).replace(
        "July", "июля"
    ).replace(
        "August", "августа"
    ).replace(
        "September", "сентября"
    ).replace(
        "October", "октября"
    ).replace(
        "November", "ноября"
    ).replace(
        "December", "декабря"
    )


def _get_weather(city: str = None) -> str:
    city = city or WEATHER_DEFAULT_CITY
    try:
        url = f"https://wttr.in/{city}?format=j1&lang=ru"
        req = urllib.request.Request(url, headers={"User-Agent": "BotGovorilka/1.0"})
        with urllib.request.urlopen(req, timeout=5) as resp:
            data = json.loads(resp.read().decode("utf-8"))
            current = data["current_condition"][0]
            temp = current["temp_C"]
            feels_like = current["FeelsLikeC"]
            desc = current["lang_ru"][0]["value"]
            humidity = current["humidity"]
            wind = current["windspeedKmph"]
            return (
                f"{city}: {desc}, {temp}°C (ощущается как {feels_like}°C). "
                f"Влажность {humidity}%, ветер {wind} км/ч."
            )
    except Exception as e:
        return f"Не удалось получить погоду для {city}: {e}"


def _set_timer(minutes: int, label: str = "Таймер") -> str:
    if minutes <= 0:
        return "Длительность таймера должна быть положительной."

    def callback():
        logger.info("⏰ Таймер '%s' (%d мин) сработал!", label, minutes)
        print(f"\n  🔔 Таймер '{label}' сработал! ({minutes} мин)")
        try:
            import winsound
            winsound.MessageBeep(winsound.MB_ICONEXCLAMATION)
        except (ImportError, RuntimeError, OSError):
            pass

    timer = threading.Timer(minutes * 60, callback)
    timer.daemon = True
    timer.start()
    return f"Таймер '{label}' установлен на {minutes} минут."


def _calculate(expression: str) -> str:
    try:
        tree = ast.parse(expression.strip(), mode="eval")
        for node in ast.walk(tree):
            if isinstance(node, (ast.Call, ast.Import, ast.ImportFrom)):
                return "Ошибка: вызовы функций и импорты запрещены."
        result = eval(compile(tree, "<calc>", "eval"))
        return f"{expression} = {result}"
    except Exception as e:
        return f"Ошибка вычисления '{expression}': {e}"


_SANDBOX_WRAPPER = """\
import ast as _ast, sys, builtins as _b

# 1. Restricted builtins — блокируем опасные функции (НО __import__ ОСТАЁТСЯ)
_BLOCKED_BUILTINS = {
    'open', 'exec', 'eval', 'compile',
    'globals', 'locals', 'vars', 'getattr', 'setattr', 'delattr',
    'breakpoint', 'exit', 'quit', 'input', 'help',
    'copyright', 'credits', 'license',
}
_safe_builtins = {k: v for k, v in vars(_b).items() if k not in _BLOCKED_BUILTINS}
_safe_builtins['__build_class__'] = _b.__build_class__
_safe_builtins['__name__'] = '__main__'

# 2. Опасные модули — полный список
_DANGEROUS_MODULES = frozenset({
    'os', 'os.path', 'posix', 'nt', 'posixpath',
    'sys', 'sysconfig',
    'subprocess', 'multiprocessing', 'threading',
    'shutil', 'pathlib', 'glob', 'fnmatch',
    'socket', 'http', 'urllib', 'ftplib', 'smtplib', 'xmlrpc',
    'ctypes', 'importlib', 'pkgutil', 'zipimport',
    'signal', 'atexit', 'webbrowser',
})

# 3. AST-валидация — блокируем опасные импорты
_src_code = sys.stdin.read()
_tree = _ast.parse(_src_code, mode='exec')

for _node in _ast.walk(_tree):
    if isinstance(_node, _ast.Import):
        for _alias in _node.names:
            _mod = _alias.name.split('.')[0]
            if _mod in _DANGEROUS_MODULES:
                raise SystemExit(f'Импорт модуля {_alias.name} запрещён в песочнице.')
    elif isinstance(_node, _ast.ImportFrom):
        if _node.module:
            _mod = _node.module.split('.')[0]
            if _mod in _DANGEROUS_MODULES:
                raise SystemExit(f'Импорт из модуля {_node.module} запрещён в песочнице.')

# 4. Оборачиваем последнее выражение в print()
_last = _tree.body[-1] if _tree.body else None
if isinstance(_last, _ast.Expr) and not (
    isinstance(_last.value, _ast.Call)
    and isinstance(_last.value.func, _ast.Name)
    and _last.value.func.id == 'print'
):
    _wrapper = _ast.Expr(value=_ast.Call(
        func=_ast.Name(id='print', ctx=_ast.Load()),
        args=[_last.value],
        keywords=[]
    ))
    _tree.body[-1] = _wrapper
    _ast.fix_missing_locations(_tree)

# 5. Запуск с ограниченными builtins
exec(compile(_tree, '<sandbox>', 'exec'), {'__builtins__': _safe_builtins})
"""


def _run_python(code: str) -> str:
    """Выполнить Python-код в изолированном процессе."""
    code = code.strip()
    if not code:
        return "Ошибка: пустой код."

    try:
        result = subprocess.run(
            [sys.executable, "-c", _SANDBOX_WRAPPER],
            input=code,
            capture_output=True,
            text=True,
            timeout=PYTHON_SANDBOX_TIMEOUT,
            creationflags=getattr(subprocess, "CREATE_NO_WINDOW", 0),
        )
    except subprocess.TimeoutExpired:
        return f"Ошибка: код выполнялся дольше {PYTHON_SANDBOX_TIMEOUT} секунд и был остановлен."
    except Exception as e:
        return f"Ошибка запуска: {e}"

    stdout = result.stdout.strip()
    stderr = result.stderr.strip()

    if result.returncode != 0:
        error_msg = stderr if stderr else "Неизвестная ошибка."
        return f"Ошибка выполнения:\n{error_msg}"

    if stdout:
        if len(stdout) > PYTHON_SANDBOX_MAX_OUTPUT:
            stdout = stdout[:PYTHON_SANDBOX_MAX_OUTPUT] + "\n... (вывод обрезан)"
        return stdout

    return "Код выполнен успешно (без вывода)."


def _search_web(query: str) -> str:
    try:
        from ddgs import DDGS

        with DDGS() as ddgs:
            results = list(ddgs.text(query, max_results=SEARCH_MAX_RESULTS))

        if not results:
            return f"Ничего не найдено по запросу '{query}'."

        parts = []
        for i, r in enumerate(results, 1):
            title = r.get("title", "")
            body = r.get("body", "")
            parts.append(f"{i}. {title}\n{body}")

        return "\n\n".join(parts)
    except ImportError:
        return "Поиск в интернете недоступен: установите duckduckgo-search."
    except Exception as e:
        return f"Ошибка поиска: {e}"


def _system_info() -> str:
    try:
        import psutil
        mem = psutil.virtual_memory()
        cpu_percent = psutil.cpu_percent(interval=0.5)
        disk = psutil.disk_usage("/")
        return (
            f"ОС: {platform.system()} {platform.release()}\n"
            f"Процессор: {platform.processor()}\n"
            f"CPU: {cpu_percent}% загрузка\n"
            f"RAM: {mem.total // (1024**3)} ГБ всего, {mem.percent}% используется\n"
            f"Диск: {disk.total // (1024**3)} ГБ всего, {disk.percent}% используется"
        )
    except ImportError:
        return (
            f"ОС: {platform.system()} {platform.release()}\n"
            f"Процессор: {platform.processor()}\n"
            f"Python: {platform.python_version()}"
        )

user_path = os.environ["USERPROFILE"].replace("\\", "/")
APP_MAP = {
    "блокнот": "notepad.exe",
    "notepad": "notepad.exe",
    "калькулятор": "calc.exe",
    "calculator": "calc.exe",
    "calc": "calc.exe",
    "проводник": "explorer.exe",
    "explorer": "explorer.exe",
    "paint": "mspaint.exe",
    "mspaint": "mspaint.exe",
    "терминал": "cmd.exe",
    "cmd": "cmd.exe",
    "powershell": "powershell.exe",
    "диспетчер задач": "taskmgr.exe",
    "task manager": "taskmgr.exe",
    "taskmgr": "taskmgr.exe",
    "настройки": "ms-settings:",
    "settings": "ms-settings:",
    "brave": "brave.exe",
    "firefox": "firefox.exe",
    "mozilla": "firefox.exe",
    "opera": "opera.exe",
    "lm studio": f"{user_path}/AppData/Local/Programs/LM Studio/LM Studio.exe",
    "лм студио": f"{user_path}/AppData/Local/Programs/LM Studio/LM Studio.exe",
    "vs code": f"{user_path}/AppData/Local/Programs/Microsoft VS Code/Code.exe",
    "вс код": f"{user_path}/AppData/Local/Programs/Microsoft VS Code/Code.exe"
    
    
}

_browser_commands = {
    "браузер": None,
    "browser": None,
    "chrome": "chrome",
    "google chrome": "chrome",
    "гугл": "chrome",
    "edge": "msedge",
    "microsoft edge": "msedge",
    "яндекс браузер": "browser",
    "yandex": "browser",
}

def _open_app(app: str) -> str:
    app_lower = app.strip().lower()
    
    # 1. Запуск стандартных утилит (Безопасно, без shell=True)
    if app_lower in APP_MAP:
        cmd = APP_MAP[app_lower]
        try:
            # Для протокола ms-settings: используем os.startfile, для .exe — Popen
            if cmd.endswith(':'):
                os.startfile(cmd)
            else:
                subprocess.Popen([cmd]) 
            return f"Приложение «{app}» запущено."
        except Exception as e:
            return f"Не удалось запустить «{app}»: {e}"
            
    # 2. Запуск браузеров (Исправлен баг со 'start')
    if app_lower in _browser_commands:
        browser = _browser_commands[app_lower]
        try:
            if browser is None:
                # Открывает дефолтный браузер системы самым безопасным путем
                webbrowser.open("https://www.google.com")
            else:
                # Запуск конкретного браузера напрямую, передавая пустую строку в аргумент
                subprocess.Popen([browser, "https://www.google.com"])
            return "Браузер открыт."
        except Exception as e:
            return f"Не удалось открыть браузер: {e}"
            
    # 3. Открытие конкретного файла по переданному пути
    if os.path.isfile(app):
        try:
            os.startfile(app)
            return f"Файл «{app}» открыт."
        except Exception as e:
            return f"Не удалось открыть «{app}»: {e}"
            
    return (
        f"Приложение «{app}» не найдено. "
        f"Доступные: блокнот, калькулятор, проводник, paint, "
        f"браузер, chrome, firefox, edge, терминал, powershell, "
        f"диспетчер задач, настройки."
    )
