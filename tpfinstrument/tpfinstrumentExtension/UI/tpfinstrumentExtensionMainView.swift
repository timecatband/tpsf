//
//  tpfinstrumentExtensionMainView.swift
//  tpfinstrumentExtension
//
//  Created by Robert Carr on 5/5/24.
//

import SwiftUI

struct tpfinstrumentExtensionMainView: View {
    var parameterTree: ObservableAUParameterGroup
    
    var body: some View {
        ParameterSlider(param: parameterTree.global.gain)
    }
}
