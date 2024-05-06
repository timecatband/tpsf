//
//  AudioUnitViewModel.swift
//  TimecatLabsSynth
//
//  Created by Robert Carr on 5/6/24.
//

import SwiftUI
import AudioToolbox
import CoreAudioKit

struct AudioUnitViewModel {
    var showAudioControls: Bool = false
    var showMIDIContols: Bool = false
    var title: String = "-"
    var message: String = "No Audio Unit loaded.."
    var viewController: ViewController?
}
