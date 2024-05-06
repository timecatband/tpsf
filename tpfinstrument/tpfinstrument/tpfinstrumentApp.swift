//
//  tpfinstrumentApp.swift
//  tpfinstrument
//
//  Created by Robert Carr on 5/5/24.
//

import CoreMIDI
import SwiftUI

@main
struct tpfinstrumentApp: App {
    @ObservedObject private var hostModel = AudioUnitHostModel()

    var body: some Scene {
        WindowGroup {
            ContentView(hostModel: hostModel)
        }
    }
}
