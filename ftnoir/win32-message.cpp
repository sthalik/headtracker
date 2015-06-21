#ifdef _WIN32
#	include <windows.h>

void process_event_loop()
{
	MSG msg;
	while(PeekMessage(&msg, NULL, 0, 0, PM_REMOVE))
	{}
}

#else
	void process_event_loop() {};
#endif