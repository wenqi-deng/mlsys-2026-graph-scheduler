#ifndef DEBUG_H_
#define DEBUG_H_
#define NDEBUG

#ifdef NDEBUG
#define DEBUG_LOG(...)                                                         \
	do {                                                                       \
	} while (0)
#else
#define DEBUG_LOG(...)                                                         \
	do {                                                                       \
		fprintf(stderr, __VA_ARGS__);                                          \
	} while (0)
#endif // NDEBUG

#endif // DEBUG_H_