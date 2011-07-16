/*
 *   ID:          $Id$
 *   Revision:    $Rev$
 */
#pragma once
#ifndef COMMON_H
#define COMMON_H

#define RAD2DEG 57.295779513082323

namespace xmatch
{
	double calc_alpha(double theta, double abs_dec);
	double calc_alpha(double theta, double zh_deg, int zone);
}

#endif /* COMMON_H */

