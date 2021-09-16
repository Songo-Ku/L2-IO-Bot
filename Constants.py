
class Constants:

    # WINDOW ELEMENT REGIONS
    SCR_SYSMSG = (20, 223, 325, 105)
    SCR_CHAT = (30,575,320,300)
    SCR_CHAR = (18,2,122,66)
    SCR_TARGET_NAME = (729,3,138,20)
    SCR_EXP = (760, 884, 51, 17)
    SCR_ADENA = (1420, 884, 93, 15)
    SCR_CPHPMP = (52, 24, 96, 64)
    SCR_CP = (56, 26, 92, 14)
    SCR_HP = (56, 38, 92, 16)
    SCR_MP = (56, 52, 92, 14)
    SCR_LVL = (21, 6, 18, 15)
    SCR_CHARNAME = (43, 4, 64,17)
    SCR_TARGNAME = (1051, 4, 139,17)
    SCR_TARGCUE = (714+1, 2+1, 12-2, 12-2)
    SCR_TARGHP = (732, 28, 150, 1)
    SCR_TARGMP = (732, 36, 150, 1)

    # multiprocess task codes
    CDE_VISION_CHARNAME = 0
    CDE_VISION_LVL = 1
    CDE_VISION_EXP = 2
    CDE_VISION_ADENA = 3
    CDE_VISION_CHAR = 4

    # CHARSETS
    CHARSET_SPECIAL = ".,/:;'\"?!@#$%^&*()-+=_><~{}[]"
    CHARSET_NUM = "0123456789"
    CHARSET_LOWER = "abcdefghijklmnopqrstuvwxyz"
    CHARSET_UPPER = "ABCDEFGHIJKLMNOPQRSTUVWXYZ"
    CHARSET_HPMP = CHARSET_NUM+"/"
    CHARSET_NAME = CHARSET_UPPER+CHARSET_LOWER+CHARSET_NUM
    CHARSET_ADENA = CHARSET_NUM+","
    CHARSET_CHAT = CHARSET_LOWER+CHARSET_UPPER+CHARSET_NUM+CHARSET_SPECIAL
    CHARSET_LVL = CHARSET_NUM

    # TEMPLATE DATASET
    TMPL_TARGET_CUE = 'target_cue'