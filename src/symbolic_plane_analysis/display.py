from rich.console import Console
from rich.layout import Layout
from rich.live import Live

console = Console(width=80, height=24)
layout = Layout()
console.print(layout)
