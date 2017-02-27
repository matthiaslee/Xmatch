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
	__host__ __device__
	double calc_alpha(double theta, double abs_dec);
	__host__ __device__
	double calc_alpha(double theta, double zh_deg, int zone);
}

#endif /* COMMON_H */

