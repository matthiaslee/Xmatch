#include "Common.h"

namespace xmatch
{
	// Gray, Nieto & Szalay 2007 (arXiv:cs/0701171v1)
	__host__ __device__
	double calc_alpha(double theta, double abs_dec)
	{
		if ( abs_dec+theta > 89.99 ) return 180;
		return abs( atan(sin(theta / RAD2DEG))
					/ sqrt( abs( cos((abs_dec-theta) / RAD2DEG) 
							   * cos((abs_dec+theta) / RAD2DEG) )) 
				);
	}

	__host__ __device__
	double calc_alpha(double theta, double zh_deg, int zone)
	{
		double dec  = abs( zone    * zh_deg - 90);
		double dec2 = abs((zone+1) * zh_deg - 90);
		dec = (dec > dec2 ? dec : dec2);
		return calc_alpha(theta, dec + 0.01*theta); // with overshoot
	}
}

