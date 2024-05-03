import mido
import math

def calculate_freq_rad_from_midi_note(note_number, sr):
    A4_MIDI_NUMBER = 69
    A4_FREQUENCY = 440.0
    freq_hz = A4_FREQUENCY * (2 ** ((note_number - A4_MIDI_NUMBER) / 12.0)) 
    return (freq_hz / sr) * 2 * math.pi 

def process_midi(midi_file_path, sr):
    midi_file = mido.MidiFile(midi_file_path)
    ticks_per_beat = midi_file.ticks_per_beat
    microseconds_per_beat = 500000  # Default tempo is 120 BPM (500,000 us per beat)

    note_events = []
    current_notes = {}  # To track note_on and their times
    time_accumulator = 0  # This will accumulate time in seconds

    for track in midi_file.tracks:
        for msg in track:
            # Handle tempo change
            if msg.type == 'set_tempo':
                microseconds_per_beat = msg.tempo
            
            # Convert time in ticks to seconds
            seconds_per_tick = (microseconds_per_beat / 1000000.0) / ticks_per_beat
            time_accumulator += msg.time * seconds_per_tick

            if msg.type == 'note_on':
                current_notes[msg.note] = time_accumulator 
            elif msg.type == 'note_off':
                if msg.note in current_notes:
                    start_time = current_notes.pop(msg.note)
                    duration = time_accumulator - start_time

                    # Conversion to samples
                    start_sample = int(start_time * sr)
                    end_sample = start_sample + int(duration * sr)  

                    freq_rad = calculate_freq_rad_from_midi_note(msg.note, sr) 
                    note_events.append((freq_rad, start_sample, end_sample))

    return note_events

if __name__ == "__main__":
    import sys
    if len(sys.argv) != 3:
        print("Usage: python parser.py <path_to_midi_file> <sample_rate>")
        sys.exit(1)
    midi_file_path = sys.argv[1]
    sr = int(sys.argv[2])
    print(process_midi(midi_file_path, sr))
