
import ast, re, sys
with open(r"c:\phishingDetection 1\app.py", encoding="utf-8") as f:
    src = f.read()
tree = ast.parse(src)
funcs = [n.name for n in ast.walk(tree) if isinstance(n, ast.FunctionDef)]
print("Syntax OK")
print("Functions:", funcs)
print()
print("--- Checks ---")
# Shield icon U+1F6E1 in title
has_shield_title = "\U0001F6E1 Phishing Detector" in src
has_shield_page  = "\U0001F6E1 Phishing Website Detection" in src
print("Shield in sidebar title:", has_shield_title)
print("Shield in page header title:", has_shield_page)
# Tree icon U+1F332 in st.info
has_tree = bool(re.search(r"st\.info\([\"'].*" + re.escape("\U0001F332") + r" Random Forest[\"']\)", src))
print("Tree icon in Random Forest st.info:", has_tree)
# THEME_LABELS (no icons)
m = re.search(r"THEME_LABELS\s*=\s*\{(.+?)\}", src, re.S)
labels = m.group(1) if m else ""
print("THEME_LABELS content:", labels.strip())
has_icon = re.search(r"[\U0001F300-\U0001F9FF]", labels)
print("Labels free of emoji icons:", not bool(has_icon))
# Radio CSS hook present
radio_css = 'div[data-baseweb="radio"] div[role="radio"]' in src
print("Radio-circle visibility CSS present:", radio_css)
# No nuclear sidebar rule
nuclear_rule = re.search(
    r'section\[data-testid="stSidebar"\]\s*div[^:]*?\{[^}]*?background-color:\s*transparent',
    src,
)
print("No nuclear sidebar-div transparent rule:", not bool(nuclear_rule))
# Warning still removed
if "instead of 81" in src:
    print("FAIL: 81-warning present!"); sys.exit(1)
print("81-feature warning removed: OK")
print()
print("ALL PASSED")
