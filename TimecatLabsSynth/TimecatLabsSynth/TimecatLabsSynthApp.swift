//
//  TimecatLabsSynthApp.swift
//  TimecatLabsSynth
//
//  Created by Robert Carr on 5/6/24.
//

import CoreMIDI
import SwiftUI

@main
struct TimecatLabsSynthApp: App {
    @ObservedObject private var hostModel = AudioUnitHostModel()

    var body: some Scene {
        WindowGroup {
            ContentView(hostModel: hostModel)
        }
    }
}
