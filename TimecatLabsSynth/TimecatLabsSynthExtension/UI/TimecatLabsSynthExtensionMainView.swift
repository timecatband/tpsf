//
//  TimecatLabsSynthExtensionMainView.swift
//  TimecatLabsSynthExtension
//
//  Created by Robert Carr on 5/6/24.
//

import SwiftUI

struct TimecatLabsSynthExtensionMainView: View {
    var parameterTree: ObservableAUParameterGroup
    var fileChosenCallback: (URL) -> Void
    var effectFileChosenCallback: (URL) -> Void
    var body: some View {
        ParameterSlider(param: parameterTree.global.gain)
        // Button to open model file chooser
        Button("Open Harmonic Model") {
            let dialog = NSOpenPanel()
            dialog.title = "Choose a model file"
            dialog.showsResizeIndicator = true
            dialog.showsHiddenFiles = false
            dialog.allowsMultipleSelection = false
            dialog.canChooseDirectories = false
            //dialog.allowedFileTypes = ["pt"]
            if dialog.runModal() == NSApplication.ModalResponse.OK {
                let result = dialog.url
                // Update audiounit parameter with new model file
                if let url = result {
                    fileChosenCallback(url)
                }

            } else {
                return
            }
        }
        Button("Open effect chain model") {
            let dialog = NSOpenPanel()
            dialog.title = "Choose a model file"
            dialog.showsResizeIndicator = true
            dialog.showsHiddenFiles = false
            dialog.allowsMultipleSelection = false
            dialog.canChooseDirectories = false
            //dialog.allowedFileTypes = ["pt"]
            if dialog.runModal() == NSApplication.ModalResponse.OK {
                let result = dialog.url
                // Update audiounit parameter with new model file
                if let url = result {
                    effectFileChosenCallback(url)
                }

            } else {
                return
            }
        }
    }
}
