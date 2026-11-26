// Intentionally omits #pragma once so macro sanitization re-applies whenever
// included after platform or C headers (<windows.h>, <wingdi.h>, <math.h>).
#ifdef OPAQUE
    #undef OPAQUE
#endif
#ifdef TRANSPARENT
    #undef TRANSPARENT
#endif
#ifdef DOMAIN
    #undef DOMAIN
#endif
#ifdef NEAR
    #undef NEAR
#endif
#ifdef FAR
    #undef FAR
#endif
#ifdef DIFFERENCE
    #undef DIFFERENCE
#endif
#ifdef ERROR
    #undef ERROR
#endif
