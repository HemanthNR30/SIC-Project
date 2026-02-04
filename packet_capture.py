from scapy.all import sniff
import time

packet_count = 0

def count_packets(pkt):
    global packet_count
    packet_count += 1

def get_packets_per_second(interface="Wi-Fi"):
    global packet_count
    packet_count = 0

    sniff(iface=interface, prn=count_packets, timeout=1, store=False)
    return packet_count
