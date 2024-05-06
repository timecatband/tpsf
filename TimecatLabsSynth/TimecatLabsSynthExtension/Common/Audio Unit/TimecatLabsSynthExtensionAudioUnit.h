//
//  TimecatLabsSynthExtensionAudioUnit.h
//  TimecatLabsSynthExtension
//
//  Created by Robert Carr on 5/6/24.
//

#import <AudioToolbox/AudioToolbox.h>
#import <AVFoundation/AVFoundation.h>

@interface TimecatLabsSynthExtensionAudioUnit : AUAudioUnit
- (void)setupParameterTree:(AUParameterTree *)parameterTree;
- (void) setModel:(NSString *)modelFile;
- (void) setEffectModel:(NSString *)modelFile;
@end
