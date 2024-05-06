//
//  TimecatLabsSynthExtensionDSPKernel.hpp
//  TimecatLabsSynthExtension
//
//  Created by Robert Carr on 5/6/24.
//

#pragma once

#import <AudioToolbox/AudioToolbox.h>
#import <CoreMIDI/CoreMIDI.h>
#import <algorithm>
#import <vector>
#import <span>
#import <string>

#import "SinOscillator.h"
#import "TimecatLabsSynthExtension-Swift.h"
#import "TimecatLabsSynthExtensionParameterAddresses.h"

#include <torch/script.h>



//class TimecatSynth {
//public:
    //TimecatSynth() {
  // Get the main bundle for the app
      //  CFBundleRef mainBundle = CFBundleGetMainBundle();
        
        // Get the URL for the model file in the bundle
//        CFURLRef modelURL = CFBundleCopyResourceURL(mainBundle, CFSTR("lms"), CFSTR("pt"), NULL);

        // Convert the URL to a regular path string
        //char modelPath[PATH_MAX];
        //CFURLGetFileSystemRepresentation(modelURL, TRUE, (UInt8*)modelPath, PATH_MAX);
            // Load the model
        //auto model = torch::jit::load(modelPath);
        //NSLog(@"Model loaded successfully");

        // Release the CFURLRef
        //if (modelURL) CFRelease(modelURL);
    //}
//};

struct ActiveNoteInfo
    {
        double velocity;
        double freqHz;
        uint32_t basePitchMidinote = 0;
        float pitchBend = 0;
        uint32_t framesRendered = 0;
        bool done = false;
        int framesSinceDone = 0;
        bool cleanup = false;
        float phase = 0;
        float lastFreqHz = 0.0;
        float lastFreqRad = 0.0;
        float phaseOffset = 0.0;
    };

#define PITCH_BEND_RANGE 48

class TimecatSynth2 {
    float mSampleRate=44100.0;
    torch::jit::script::Module model;
    torch::jit::script::Module effect_chain;
    torch::Tensor fadeWindow = torch::linspace(1, 0, 256);
    bool hasEffectChain = false;
    public:
    TimecatSynth2(std::string model_file_url) {
        // Load the model
        model = torch::jit::load(model_file_url);
        NSLog(@"Model loaded successfully from chosen path");
        // Disable grad globally

    }
    void loadEffectChain(std::string model_file_url) {
        NSLog(@"Loading effect chain from path");
        try {
        effect_chain = torch::jit::load(model_file_url);
        hasEffectChain = true;
        } catch (const std::exception& e) {
            NSLog(@"Error loading effect chain");
            NSLog(@"%s", e.what());
        }
        NSLog(@"Effect chain loaded successfully from chosen path");
    }
    void renderNote(ActiveNoteInfo &note, std::span<float *> outputBuffers, AUAudioFrameCount frameCount) {
        torch::NoGradGuard no_grad;
        float t_start = note.framesRendered / (float)mSampleRate;
        float t_end = (note.framesRendered + frameCount) / (float)mSampleRate;
        float step = (t_end - t_start) / (float)frameCount;
        auto t = torch::arange(t_start, t_end, step);
        // Print T shape
        // Calculate freqRad using sample rate
        // Interpolate freq hz using pitch bend
        auto freqHz = note.freqHz * (1.0f+note.pitchBend*7.0f);

        auto origRad = 2 * M_PI * note.freqHz;
        auto freqRad = 2 * M_PI * freqHz;
    
        // Interpolate between freqRad and 8* freqRad according to pitchBend
       // freqRad = freqRad + (8 * freqRad - freqRad) * note.pitchBend;

        auto realRad = torch::tensor({origRad / mSampleRate});
        auto time_latent = torch::tensor({1.4,-.1});
        auto amplitude = torch::tensor({note.velocity});
        // Use the original to avoid weird harmonic scaling
        auto hamps = model.forward({realRad, time_latent, amplitude}).toTensor();
        // Generate a sin wave
        auto freqRadTensor = torch::full({t.size(0)}, freqRad);

         // If pitch bending, interpolate between last and new frequency radian value
        if (note.lastFreqRad != 0.0) {
            auto lastFreqRadTensor = torch::full({t.size(0)}, note.lastFreqRad);
            auto fadeIn = torch::linspace(0, 1, t.size(0), torch::kFloat32);
            freqRadTensor = lastFreqRadTensor * (1 - fadeIn) + freqRadTensor * fadeIn;
        }
        // TODO: Why doesn't it matter that this sample rate is wrong?
        freqRadTensor = freqRadTensor / mSampleRate;
        auto phase = torch::cumsum(freqRadTensor, 0) + note.phaseOffset;
        
        note.lastFreqRad = freqRad;
        auto sin_wave = torch::sin(phase)*hamps[0][0];
        for (int i = 1; i < hamps.size(1); i++) {
            if (freqHz * (i+1) > 20000) {
                break;
            }
            sin_wave += hamps[0][0]*torch::sin((i+1)*phase)*hamps[0][i];
        }
        // TODO: Must fix
        float max = torch::sum(torch::abs(hamps)).item<float>();
        sin_wave = sin_wave / max;
        if (hasEffectChain) {
            auto t_start_tensor = torch::tensor({t_start});
            auto effect_output = effect_chain.forward({sin_wave, t_start_tensor});
            sin_wave = effect_output.toTensor();
        }
        auto maxhamps = torch::max(hamps);
        //NSLog(@"Max hamps: %f", maxhamps.item<float>());
    
        // Print shape
                // Generate per sample dsp before assigning it to out
        for (UInt32 frameIndex = 0; frameIndex < frameCount; ++frameIndex) {
            // Do your frame by frame dsp here...
            auto sample = sin_wave[frameIndex].item<float>();
            for (UInt32 channel = 0; channel < outputBuffers.size(); ++channel) {
                if (note.done) {
                    sample *= 1 - (note.framesSinceDone / 512.0);
                    if (note.framesSinceDone >= 512) {
                        note.cleanup = true;
                    } else {
                        note.framesSinceDone += 1;
                    }

                }
                outputBuffers[channel][frameIndex] += sample;
            }
        }

        note.framesRendered += frameCount;
        note.phaseOffset = phase[-1].item<float>();
    }
};

/*
 TimecatLabsSynthExtensionDSPKernel
 As a non-ObjC class, this is safe to use from render thread.
 */
 #define NUM_MIDI_CHANNELS 16
class TimecatLabsSynthExtensionDSPKernel {
    std::vector<ActiveNoteInfo> mActiveNotes;
    std::vector<ActiveNoteInfo> activeNotesPerChannel[NUM_MIDI_CHANNELS];
    TimecatSynth2 *mSynth = nullptr;
public:
    void initialize(int channelCount, double inSampleRate) {
        mSampleRate = inSampleRate;
        mSinOsc = SinOscillator(inSampleRate);
        NSLog(@"Initializing TimecatSynth");
      //  mSynth = new TimecatSynth();
        NSLog(@"TimecatSynth initialized");
    }
    void setModel(NSString* model_file_url) {
        NSLog(@"Setting model");
        std::string model_file_url_std = [model_file_url UTF8String];
        mSynth = new TimecatSynth2(model_file_url_std);
        NSLog(@"Model set");
    }
    
    void setEffectModel(NSString* model_file_url) {
        NSLog(@"Setting effect model from path %s", [model_file_url UTF8String]);
        std::string model_file_url_std = [model_file_url UTF8String];
        mSynth->loadEffectChain(model_file_url_std);
    }

    void deInitialize() {
    }
    
    // MARK: - Bypass
    bool isBypassed() {
        return mBypassed;
    }
    
    void setBypass(bool shouldBypass) {
        mBypassed = shouldBypass;
    }
    
    // MARK: - Parameter Getter / Setter
    // Add a case for each parameter in TimecatLabsSynthExtensionParameterAddresses.h
    void setParameter(AUParameterAddress address, AUValue value) {
        switch (address) {
            case TimecatLabsSynthExtensionParameterAddress::gain:
                mGain = value;
                break;
        }
    }
    
    AUValue getParameter(AUParameterAddress address) {
        // Return the goal. It is not thread safe to return the ramping value.
        
        switch (address) {
            case TimecatLabsSynthExtensionParameterAddress::gain:
                return (AUValue)mGain;
                
            default: return 0.f;
        }
    }
    
    // MARK: - Max Frames
    AUAudioFrameCount maximumFramesToRender() const {
        return mMaxFramesToRender;
    }
    
    void setMaximumFramesToRender(const AUAudioFrameCount &maxFrames) {
        mMaxFramesToRender = maxFrames;
    }
    
    // MARK: - Musical Context
    void setMusicalContextBlock(AUHostMusicalContextBlock contextBlock) {
        mMusicalContextBlock = contextBlock;
    }
    
    // MARK: - MIDI Protocol
    MIDIProtocolID AudioUnitMIDIProtocol() const {
        return kMIDIProtocol_2_0;
    }
    
    inline double MIDINoteToFrequency(int note) {
        constexpr auto kMiddleA = 440.0;
        return (kMiddleA / 32.0) * pow(2, ((note - 9) / 12.0));
    }
    
    /**
     MARK: - Internal Process
     
     This function does the core siginal processing.
     Do your custom DSP here.
     */
    void process(std::span<float *> outputBuffers, AUEventSampleTime bufferStartTime, AUAudioFrameCount frameCount) {
        //if (mBypassed || mSynth == nullptr) {
            // Fill the 'outputBuffers' with silence
            for (UInt32 channel = 0; channel < outputBuffers.size(); ++channel) {
                std::fill_n(outputBuffers[channel], frameCount, 0.f);
            }
           // if (mBypassed || mSynth == nullptr) {
             //   NSLog(@"Bypassed or synth not initialized");
               // return;
            //}
        //}
        
        // Use this to get Musical context info from the Plugin Host,
        // Replace nullptr with &memberVariable according to the AUHostMusicalContextBlock function signature
        if (mMusicalContextBlock) {
            mMusicalContextBlock(nullptr /* currentTempo */,
                                 nullptr /* timeSignatureNumerator */,
                                 nullptr /* timeSignatureDenominator */,
                                 nullptr /* currentBeatPosition */,
                                 nullptr /* sampleOffsetToNextBeat */,
                                 nullptr /* currentMeasureDownbeatPosition */);
        }
        
        // Generate per sample dsp before assigning it to out
    //    for (UInt32 frameIndex = 0; frameIndex < frameCount; ++frameIndex) {
            // Do your frame by frame dsp here...
      //      const auto sample = mSinOsc.process() * mNoteEnvelope * mGain;

        //    for (UInt32 channel = 0; channel < outputBuffers.size(); ++channel) {
          //      outputBuffers[channel][frameIndex] = sample;
            //}
        //}
        //auto ones = torch::ones({1});
      

        for (int i = 0; i < NUM_MIDI_CHANNELS; i++) {
            activeNotesPerChannel[i].erase(std::remove_if(activeNotesPerChannel[i].begin(), activeNotesPerChannel[i].end(), [&](const ActiveNoteInfo& info) {
                return info.cleanup == true;
            }), activeNotesPerChannel[i].end());
        }
        for (int i = 0; i < NUM_MIDI_CHANNELS; i++) {
            for (ActiveNoteInfo& note : activeNotesPerChannel[i]) {
                try {
                mSynth->renderNote(note, outputBuffers, frameCount);
                } catch (const std::exception& e) {
                    NSLog(@"Error rendering note");
                    NSLog(@"%s", e.what());
                }
            }
        }
    }
    
    void handleOneEvent(AUEventSampleTime now, AURenderEvent const *event) {
        NSLog(@"Handling event");
        switch (event->head.eventType) {
            case AURenderEventParameter: {
                handleParameterEvent(now, event->parameter);
                break;
            }
                
            case AURenderEventMIDIEventList: {
                handleMIDIEventList(now, &event->MIDIEventsList);
                break;
            }
                
            default:
                break;
        }
    }
    
    void handleParameterEvent(AUEventSampleTime now, AUParameterEvent const& parameterEvent) {
        // Implement handling incoming Parameter events as needed
    }
    
    void handleMIDIEventList(AUEventSampleTime now, AUMIDIEventList const* midiEvent) {
        auto visitor = [] (void* context, MIDITimeStamp timeStamp, MIDIUniversalMessage message) {
            auto thisObject = static_cast<TimecatLabsSynthExtensionDSPKernel *>(context);
            
            switch (message.type) {
                case kMIDIMessageTypeChannelVoice2: {
                    thisObject->handleMIDI2VoiceMessage(message);
                }
                    break;
                    
                default:
                    break;
            }
        };
        
        MIDIEventListForEachEvent(&midiEvent->eventList, visitor, this);
    }
    
    void handleMIDI2VoiceMessage(const struct MIDIUniversalMessage& message) {
        const auto& note = message.channelVoice2.note;
        auto channel = message.channelVoice2.channel;
        if (channel > NUM_MIDI_CHANNELS) {
            NSLog(@"Invalid midi channel %d", channel);
                        channel = 0;
        }
        auto& activeNotes = activeNotesPerChannel[channel];

        
        switch (message.channelVoice2.status) {
            case kMIDICVStatusNoteOff: {
                mNoteEnvelope = 0.0;
                auto freq = MIDINoteToFrequency(note.number);
             //   mActiveNotes.erase(std::remove_if(mActiveNotes.begin(), mActiveNotes.end(), [&](const ActiveNoteInfo& info) {
              //      return MIDINoteToFrequency(note.number) == info.freqHz;
               // }), mActiveNotes.end());
               for (auto it = activeNotes.begin(); it != activeNotes.end(); ) {
                   if (it->freqHz == freq) {
                       it->done = true;
                       it->framesSinceDone = 0;
                   } 
                   ++it;
               }
                NSLog(@"Note off");
            }
            break;
                
            case kMIDICVStatusNoteOn: {
                NSLog(@"Note on");
                const auto velocity = message.channelVoice2.note.velocity;
                const auto freqHertz   = MIDINoteToFrequency(note.number);

                ActiveNoteInfo info = ActiveNoteInfo{velocity/127.0, freqHertz, note.number};
                activeNotes.push_back(info);
                // Use velocity to set amp envelope level
                //mNoteEnvelope = (double)velocity / (double)std::numeric_limits<std::uint16_t>::max();
            }
            break;
            case kMIDICVStatusPitchBend: {
                for (auto& note : activeNotes) {
                      UInt32 bendData = message.channelVoice2.pitchBend.data;
                      uint64_t center_point = 2147483648; // use uint64_t to handle larger calculations
                      int64_t normalizedBend = (int64_t)bendData - center_point; // explicit cast to handle negative values properly
                      note.pitchBend = (float)normalizedBend / (float)(center_point * 2); // correctly handles the 64-bit result of 2*center_point
                }
            }
            break;
                
            default:
                break;
        }
    }
    // MARK: - Member Variables
    AUHostMusicalContextBlock mMusicalContextBlock;
    
    double mSampleRate = 44100.0;
    double mGain = 1.0;
    double mNoteEnvelope = 0.0;
    
    bool mBypassed = false;
    AUAudioFrameCount mMaxFramesToRender = 1024;
    
    SinOscillator mSinOsc;
};
