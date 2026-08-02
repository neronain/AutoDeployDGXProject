"""ทำให้ tests/ เป็น package — เทสหลายไฟล์ import helper กันเอง (`from tests.test_generator import ...`)

ถ้าไม่มีไฟล์นี้ pytest จะใส่ tests/ ลง sys.path แทน repo root ทำให้ `import tests.*`
ล้มด้วย ModuleNotFoundError เวลารัน `pytest` ตรง ๆ (บังเอิญผ่านเฉพาะตอนรัน `python -m pytest`
เพราะโหมดนั้นใส่ cwd ให้เอง)
"""
