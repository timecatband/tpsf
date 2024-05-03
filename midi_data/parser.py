import mido
from mido.messages.messages import Message
import math

def calculate_freq_rad_from_midi_note(note_number, sr):
    A4_MIDI_NUMBER = 69
    A4_FREQUENCY = 440.0
    freq_hz = A4_FREQUENCY * (2 ** ((note_number - A4_MIDI_NUMBER) / 12.0)) 
    return (freq_hz / sr) * 2 * math.pi 

def process_midi(midi_file_path, sr):
    midi_file = mido.MidiFile(midi_file_path)
    ticks_per_beat = midi_file.ticks_per_beat

    note_events = []
    current_notes = {}  # To track note_on and their times

    time_accumulator = 0
    for track in midi_file.tracks:
        for msg in track:
            time_accumulator += msg.time

            if not isinstance(msg, Message):  
                continue  

            if msg.type == 'note_on':
                current_notes[msg.note] = time_accumulator 
            elif msg.type == 'note_off':
                if msg.note in current_notes:
                    start_time = current_notes.pop(msg.note)
                    duration = time_accumulator - start_time

                    # Conversion to samples (you'll need your sample rate 'sr')
                    start_sample = int(start_time * sr / ticks_per_beat)
                    end_sample = start_sample + int(duration * sr / ticks_per_beat)  

                    freq_rad = calculate_freq_rad_from_midi_note(msg.note, sr) 
                    note_events.append((freq_rad, start_sample, end_sample))  # Updated tuple

    return note_events

import sys
if __name__ == "__main__":
    if len(sys.argv) != 3:
        print("Usage: python parser.py <path_to_midi_file> <sample_rate>")
        sys.exit(1)
    midi_file_path = sys.argv[1]
    sr = int(sys.argv[2])
    print(process_midi(midi_file_path, sr))
