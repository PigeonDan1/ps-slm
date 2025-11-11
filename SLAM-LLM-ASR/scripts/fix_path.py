import sys, site
try:
    user_site = site.getusersitepackages()
    if user_site in sys.path:
        sys.path.remove(user_site)
except Exception:
    pass