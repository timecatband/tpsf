//
//  TimecatLabsSynthExtensionParameterAddresses.h
//  TimecatLabsSynthExtension
//
//  Created by Robert Carr on 5/6/24.
//

#pragma once

#include <AudioToolbox/AUParameters.h>

#ifdef __cplusplus
namespace TimecatLabsSynthExtensionParameterAddress {
#endif

typedef NS_ENUM(AUParameterAddress, TimecatLabsSynthExtensionParameterAddress) {
    gain = 0,
    model_file = 1,
};

#ifdef __cplusplus
}
#endif
