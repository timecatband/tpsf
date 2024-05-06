//
//  tpfinstrumentExtensionAudioUnit.h
//  tpfinstrumentExtension
//
//  Created by Robert Carr on 5/5/24.
//

#import <AudioToolbox/AudioToolbox.h>
#import <AVFoundation/AVFoundation.h>

@interface tpfinstrumentExtensionAudioUnit : AUAudioUnit
- (void)setupParameterTree:(AUParameterTree *)parameterTree;
@end
