import os
from smartmoneyconcepts.smc import smc

if os.getenv('SMC_CREDIT', '1') == '1':
    print("\033[1;33mThank you for using SmartMoneyConcepts! ⭐\033[0m")