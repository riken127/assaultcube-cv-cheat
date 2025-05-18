import time
import csv
import ctypes
from ctypes import wintypes
import socket

# Offsets
LOCAL_PLAYER_OFFSET = 0x0017E0A8  # Offset relativo ao módulo base
HEALTH_OFFSET = 0xEC
PRIMARY_AMMO_OFFSET = 0x140
PRIMARY_IN_MAG_OFFSET = 0x11C
SECONDARY_AMMO_OFFSET = 0x12C
SECONDARY_IN_MAG_OFFSET = 0x108

PROCESS_ALL_ACCESS = 0x1F0FFF

# Estrutura  (simplificada para 64-bit segura)
class MODULEENTRY32(ctypes.Structure):
    _fields_ = [
        ('dwSize', wintypes.DWORD),
        ('th32ModuleID', wintypes.DWORD),
        ('th32ProcessID', wintypes.DWORD),
        ('GlblcntUsage', wintypes.DWORD),
        ('ProccntUsage', wintypes.DWORD),
        ('modBaseAddr', ctypes.POINTER(ctypes.c_byte)),
        ('modBaseSize', wintypes.DWORD),
        ('hModule', wintypes.HMODULE),
        ('szModule', ctypes.c_char * 256),
        ('szExePath', ctypes.c_char * wintypes.MAX_PATH)
    ]

def get_process_handle_and_pid(process_name="ac_client.exe"):
    kernel32 = ctypes.WinDLL("kernel32", use_last_error=True)
    CreateToolhelp32Snapshot = kernel32.CreateToolhelp32Snapshot
    Process32First = kernel32.Process32First
    Process32Next = kernel32.Process32Next

    class PROCESSENTRY32(ctypes.Structure):
        _fields_ = [
            ("dwSize", wintypes.DWORD),
            ("cntUsage", wintypes.DWORD),
            ("th32ProcessID", wintypes.DWORD),
            ("th32DefaultHeapID", ctypes.POINTER(ctypes.c_ulong)),
            ("th32ModuleID", wintypes.DWORD),
            ("cntThreads", wintypes.DWORD),
            ("th32ParentProcessID", wintypes.DWORD),
            ("pcPriClassBase", ctypes.c_long),
            ("dwFlags", wintypes.DWORD),
            ("szExeFile", ctypes.c_char * wintypes.MAX_PATH),
        ]

    snapshot = CreateToolhelp32Snapshot(0x00000002, 0)  # TH32CS_SNAPPROCESS
    entry = PROCESSENTRY32()
    entry.dwSize = ctypes.sizeof(PROCESSENTRY32)

    if not Process32First(snapshot, ctypes.byref(entry)):
        raise Exception("Process32First failed")

    while True:
        if entry.szExeFile.decode() == process_name:
            pid = entry.th32ProcessID
            handle = kernel32.OpenProcess(PROCESS_ALL_ACCESS, False, pid)
            kernel32.CloseHandle(snapshot)
            return handle, pid
        if not Process32Next(snapshot, ctypes.byref(entry)):
            break

    kernel32.CloseHandle(snapshot)
    raise Exception(f"Process '{process_name}' not found.")

def get_module_base_address(pid, module_name="ac_client.exe"):
    kernel32 = ctypes.WinDLL("kernel32", use_last_error=True)
    CreateToolhelp32Snapshot = kernel32.CreateToolhelp32Snapshot
    Module32First = kernel32.Module32First
    Module32Next = kernel32.Module32Next

    snapshot = CreateToolhelp32Snapshot(0x00000008, pid)  # TH32CS_SNAPMODULE
    module_entry = MODULEENTRY32()
    module_entry.dwSize = ctypes.sizeof(MODULEENTRY32)

    if not Module32First(snapshot, ctypes.byref(module_entry)):
        raise Exception("Module32First failed")

    while True:
        if module_entry.szModule.decode() == module_name:
            base_address = ctypes.addressof(module_entry.modBaseAddr.contents)
            kernel32.CloseHandle(snapshot)
            return base_address
        if not Module32Next(snapshot, ctypes.byref(module_entry)):
            break

    kernel32.CloseHandle(snapshot)
    raise Exception(f"Module '{module_name}' not found.")

def read_memory(process_handle, address, size=4):
    buffer = ctypes.create_string_buffer(size)
    bytes_read = ctypes.c_size_t()
    ctypes.windll.kernel32.ReadProcessMemory(process_handle, ctypes.c_void_p(address), buffer, size, ctypes.byref(bytes_read))
    return int.from_bytes(buffer.raw, byteorder="little")

#def log_to_csv(data, log_file="memory_log.csv"):
#    with open(log_file, "a", newline="") as f:
#        writer = csv.writer(f)
#        writer.writerow(data)

def main():
    try:
        process_handle, pid = get_process_handle_and_pid()
        base_address = get_module_base_address(pid)
        local_player_ptr = read_memory(process_handle, base_address + LOCAL_PLAYER_OFFSET)

        sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
        server_address = ("127.0.0.1", 5005)

        while True:
            health = read_memory(process_handle, local_player_ptr + HEALTH_OFFSET)
            primary_ammo = read_memory(process_handle, local_player_ptr + PRIMARY_AMMO_OFFSET)
            primary_in_mag = read_memory(process_handle, local_player_ptr + PRIMARY_IN_MAG_OFFSET)
            secondary_ammo = read_memory(process_handle, local_player_ptr + SECONDARY_AMMO_OFFSET)
            secondary_in_mag = read_memory(process_handle, local_player_ptr + SECONDARY_IN_MAG_OFFSET)

            log_entry = [
                time.time(),
                health,
                primary_ammo,
                primary_in_mag,
                secondary_ammo,
                secondary_in_mag
            ]
#            log_to_csv(log_entry)
            sock.sendto(str(log_entry).encode(), server_address)

            time.sleep(0.2)
    except Exception as e:
        print(f"[ERRO] {e}")

if __name__ == "__main__":
    main()
