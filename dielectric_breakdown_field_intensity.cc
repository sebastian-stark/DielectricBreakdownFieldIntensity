// --------------------------------------------------------------------------
// Copyright (C) 2021 by Sebastian Stark
//
// This program is free software; you can redistribute it and/or
// modify it under the terms of the GNU General Public License
// as published by the Free Software Foundation; either version 2
// of the License, or (at your option) any later version.
//
// This program is distributed in the hope that it will be useful,
// but WITHOUT ANY WARRANTY; without even the implied warranty of
// MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
// GNU General Public License for more details.
//
// You should have received a copy of the GNU General Public License
// along with this program; if not, write to the Free Software
// Foundation, Inc., 51 Franklin Street, Fifth Floor, Boston, MA  02110-1301, USA.

#include <iostream>
#include <fstream>
#include <math.h>

#include <deal.II/fe/fe_q.h>
#include <deal.II/grid/grid_generator.h>
#include <deal.II/grid/grid_tools.h>

#include <incremental_fe/fe_model.h>


using namespace std;
using namespace dealii;
using namespace dealii::GalerkinTools;
using namespace incrementalFE;

/**
 * class defining the dielectric energy density
 */
template<unsigned int spacedim>
class DielectricEnergy : public ScalarFunctional<spacedim, spacedim>
{

private:

	/**
	 * permittivity
	 */
	const double eps;

public:

	/**
	 * The constructor of the class.
	 *
	 * @param[in]	e_omega					ScalarFunctional<spacedim, spacedim>::e_omega
	 *
	 * @param[in]	domain_of_integration	ScalarFunctional<spacedim, spacedim>::domain_of_integration
	 *
	 * @param[in]	quadrature				ScalarFunctional<spacedim, spacedim>::quadrature
	 *
	 * @param[in]	eps						DielectricEnergy::eps
	 *
	 * @param[in]	name					ScalarFunctional<spacedim, spacedim>::name
	 */
	DielectricEnergy(	const std::vector<DependentField<spacedim,spacedim>>	e_omega,
						const std::set<types::material_id>						domain_of_integration,
						const Quadrature<spacedim>								quadrature,
						const double											eps,
						const std::string										name = "DielectricEnergy")
	:
	ScalarFunctional<spacedim, spacedim>(e_omega, domain_of_integration, quadrature, name, 0),
	eps(eps)
	{}

	/**
	 * See ScalarFunctional<spacedim, spacedim>::get_h_omega for detailed information on this method.
	 */
	bool
	get_h_omega(Vector<double>& 					e_omega,
				const std::vector<Vector<double>>&	/*e_omega_ref_sets*/,
				Vector<double>&						/*hidden_vars*/,
				const Point<spacedim>&				x,
				double&								h_omega,
				Vector<double>&						h_omega_1,
				FullMatrix<double>&					h_omega_2,
				const std::tuple<bool, bool, bool>	requested_quantities)
	const
	{
		Assert(e_omega.size() == this->e_omega.size(),ExcMessage("Called get_h_omega with invalid size of e_omega vector!"));

		Tensor<1,spacedim> E;
		E[0] = e_omega[0];
		E[1] = e_omega[1];

		if(get<0>(requested_quantities) )
		{
			h_omega = eps * numbers::PI * E * E * x[0];
		}

		if(get<1>(requested_quantities))
		{
			if(h_omega_1.size() != this->e_omega.size())
				h_omega_1.reinit(this->e_omega.size());
			h_omega_1[0] = eps * 2.0 * numbers::PI * E[0] * x[0];
			h_omega_1[1] = eps * 2.0 * numbers::PI * E[1] * x[0];
		}

		if(get<2>(requested_quantities))
		{
			if( (h_omega_2.size()[0]!=this->e_omega.size()) || (h_omega_2.size()[1]!=this->e_omega.size()) )
				h_omega_2.reinit(this->e_omega.size(), this->e_omega.size());
			h_omega_2(0,0) = 2.0 * numbers::PI * eps * x[0];
			h_omega_2(1,1) = 2.0 * numbers::PI * eps * x[0];
		}

		return false;
	}
};


/**
 * @param[in]	d		thickness of capacitor
 *
 * @param[in]	N_ref	number of global mesh_refinements
 *
 * @param[in]	shift	shift of top electrode position in radial direction
 *
 * @return				potential
 */
double
compute_potential(	const double 		d,
					const unsigned int	N_ref,
					const unsigned int 	N_R,
					const double		shift)
{

/********************
 * parameters, etc. *
 ********************/

	const unsigned int spacedim = 2;	// spatial dimensionality; this must be spacedim = 2
	const double R = 15.0; 				// sample radius (mm)
	const double R_el_1 = 5.0;			// radius of top electrode (mm)
	const double R_el_2 = 12.5;			// radius of bottom electrode (mm)
	const double eps_i = 10.0;			// relative permittivity of alumina ceramics
	const double eps_l = 3.2;			// relative permittivity of MIDEL7131
	const unsigned int N_e_R = 6;		// number of (equally spaced) finite elements to mesh sample in radial direction,
										// warning: element size l_e = R/N_e_R must be such that R_el_1/l_e and R_el_2/l_e are integers (otherwise the electrode edge positions will not be accurate)
	const unsigned int N_ref_edge = 4;  // extra refinements at edge
	const unsigned int degree = 2;		// finite element degree

	// mappings
	MappingQGeneric<spacedim, spacedim> mapping_domain(degree);					// FE mapping on domain
	MappingQGeneric<spacedim-1, spacedim> mapping_interface(degree);			// FE mapping on interfaces


/********
 * grid *
 ********/

	// make domain triangulation
	Triangulation<spacedim> tria_domain;
	const unsigned int N_e = N_e_R * N_R;
	const double l_e = R / (double)N_e_R;
	const double H = (double)N_e * l_e;
	const double R_el_1_shifted = R_el_1 + shift;
	GridGenerator::subdivided_hyper_rectangle(tria_domain, {N_e, 2 * N_e}, Point<spacedim>(0.0, -H), Point<spacedim>(H, H));
	const auto transform_1 = [&](const Point<spacedim>& point) -> Point<spacedim>
	{
		const double x = point[0];
		const double y = point[1];
		auto ret = point;
		ret[0] = x < R_el_2 ? R_el_2 * ( (-x) * (R_el_1-x)) / ( (-R_el_2) * (R_el_1-R_el_2) ) + R_el_1_shifted * ( (-x) * (R_el_2-x)) / ( (-R_el_1) * (R_el_2-R_el_1) ) : x;
		ret[1] = y > 0 ? H * ( (-y) * (l_e-y)) / ( (-H) * (l_e-H) ) + d * ( (-y) * (H-y)) / ( (-l_e) * (H-l_e) ) : -H * ( (y) * (l_e+y)) / ( -H * (l_e-H) ) - d * ( y * (H+y)) / ( -l_e * (H-l_e) );
		return ret;
	};
	GridTools::transform(transform_1, tria_domain);

	// triangulation system
	dealii::GalerkinTools::TriangulationSystem<spacedim> tria_system(tria_domain);

	// define material regions
	for(const auto& cell : tria_domain.cell_iterators_on_level(0))
	{
		if( (cell->center()[0] < R) && (cell->center()[1] < d) && (cell->center()[1] > 0))
			cell->set_material_id(0);
		else
			cell->set_material_id(1);
	}

	// define electrodes
	for(const auto& cell : tria_domain.cell_iterators_on_level(0))
	{
		if(cell->material_id() == 0)
		{
			for(unsigned int face = 0; face < GeometryInfo<spacedim>::faces_per_cell; ++face)
			{
				if( (cell->face(face)->center()[1] < 1e-10) && (cell->face(face)->center()[0] < R_el_2) )
					tria_system.add_interface_cell(cell, face, 2);
				else if( ( fabs(cell->face(face)->center()[1] - d) < 1e-12) && (cell->face(face)->center()[0] < R_el_1_shifted) )
					tria_system.add_interface_cell(cell, face, 1);
			}
		}
	}
	tria_system.close();

	// initial mesh refinement at electrode edges
	const unsigned int N_init = (double)(log(l_e/d) / log(2.0)) + 1 + N_ref_edge;
	const auto p_edge_1 = Point<spacedim>(R_el_1_shifted, d);
	const auto p_edge_2 = Point<spacedim>(R_el_2, 0.0);
	for(unsigned int m = 0; m < N_init; ++m)
	{
		for(const auto& cell : tria_domain.active_cell_iterators())
		{
			for(unsigned int v = 0; v < GeometryInfo<spacedim>::vertices_per_cell; ++v)
			{
				if( (cell->vertex(v).distance(p_edge_1) < 1e-12) || (cell->vertex(v).distance(p_edge_2) < 1e-12) )
					cell->set_refine_flag();
			}
		}
		tria_domain.execute_coarsening_and_refinement();
	}
	for(const auto& cell : tria_domain.active_cell_iterators())
	{
		for(unsigned int f = 0; f < GeometryInfo<spacedim>::faces_per_cell; ++f)
		{
			if(!cell->face(f)->at_boundary())
			{
				if(cell->material_id() != cell->neighbor(f)->material_id())
				{
					if(cell->level() < cell->neighbor(f)->level())
					{
						cell->set_refine_flag();
					}
					else if(cell->level() > cell->neighbor(f)->level())
					{
						cell->neighbor(f)->set_refine_flag();
					}
				}
			}
		}
	}
	tria_domain.execute_coarsening_and_refinement();


	// global mesh refinement
	tria_domain.refine_global(N_ref);

	// write mesh to files
	tria_system.write_triangulations_vtk("tria_domain.vtk", "tria_interface.vtk");

/**********************
 * independent fields *
 **********************/

	IndependentField<spacedim, spacedim> phi("phi", FE_Q<spacedim>(degree), 1, {0,1});	// scalar potential

	Functions::ConstantFunction<spacedim> constant_fun(1.0);
	DirichletConstraint<spacedim> dc_phi_2(phi, 0, InterfaceSide::minus, {2});					// constraint bottom electrode
	DirichletConstraint<spacedim> dc_phi_1(phi, 0, InterfaceSide::minus, {1}, &constant_fun);	// constraint bottom electrode

	Constraints<spacedim> constraints;
	constraints.add_dirichlet_constraint(dc_phi_2);
	constraints.add_dirichlet_constraint(dc_phi_1);

/********************
 * dependent fields *
 ********************/

	// electric fields
	DependentField<spacedim, spacedim> E_r("E_r");
	E_r.add_term(-1.0, phi, 0, 0);
	DependentField<spacedim, spacedim> E_z("E_z");
	E_z.add_term(-1.0, phi, 0, 1);

	// scalar potential on interface
	DependentField<spacedim-1, spacedim> phi_S("phi_S");
	phi_S.add_term(1.0, phi, 0, InterfaceSide::minus);

/*************************
 * incremental potential *
 *************************/

	DielectricEnergy<spacedim> psi_i(	{E_r, E_z},
										{0},
										QGauss<spacedim>(degree+1),
										eps_i);

	DielectricEnergy<spacedim> psi_l(	{E_r, E_z},
										{1},
										QGauss<spacedim>(degree+1),
										eps_l);

	TotalPotentialContribution<spacedim> psi_i_tpc(psi_i);
	TotalPotentialContribution<spacedim> psi_l_tpc(psi_l);

	TotalPotential<spacedim> total_potential;
	total_potential.add_total_potential_contribution(psi_i_tpc);
	total_potential.add_total_potential_contribution(psi_l_tpc);

/***************************
 * Solution of the problem *
 ***************************/

	BlockSolverWrapperMUMPS solver_wrapper_mumps;
	solver_wrapper_mumps.icntl[2] = 6;		// standard output stream
	solver_wrapper_mumps.icntl[3] = 1;		// print level
	solver_wrapper_mumps.icntl[7] = 0;		// row scaling
	solver_wrapper_mumps.icntl[6] = 5;		// use METIS ordering
	solver_wrapper_mumps.icntl[27] = 1;		// sequential calculation
	solver_wrapper_mumps.analyze = 1;		// only analyze matrix structure in first step as it does not change subsequently

	// set up finite element model
	GlobalDataIncrementalFE<spacedim> global_data;
	FEModel<spacedim, Vector<double>, BlockVector<double>, GalerkinTools::TwoBlockMatrix<SparseMatrix<double>>> fe_model(total_potential, tria_system, mapping_domain, mapping_interface, global_data, constraints, solver_wrapper_mumps, true, true);

	// solve
	fe_model.do_time_step(1.0);

	// write output
	fe_model.write_output_independent_fields("results/output_domain", "results/output_interface", degree);
	const double potential_value = fe_model.get_potential_value();
	return potential_value / (0.5 * eps_i / d);

}


int main(int argc, char **argv)
{

	Utilities::MPI::MPI_InitFinalize mpi_initialization(argc, argv, 1);

	const double d_min = 0.2;						// minimum thickness of capacitor
	const double d_max = 2.0;						// maximum thickness of capacitor
	const double steps = 20;						// number of thickness steps
	const unsigned int N_ref_max = 4;				// global mesh refinements
	const vector<unsigned int> N_R_vect = {2, 4, 8};// radii of the entire computational domain in multiples of sample radius to try


	for(const auto N_R : N_R_vect)
	{
		for(unsigned int N_ref = 0; N_ref <= N_ref_max; ++N_ref)
		{
			const string file_name_res	= "results/results_N_ref=" + Utilities::to_string(N_ref) + "_N_R=" + Utilities::to_string(N_R) + ".dat";				// file where results are stored

			vector<pair<double,double>> G_vect;
			for(unsigned int m = 0; m < steps; ++m)
			{
				const double d = d_min + (d_max - d_min) / (double)(steps - 1) * (double)m;
				const double R_el_1 = 5.0;
				const double dR_el_1 = 0.01 * d / pow(2, (double)N_ref);
				const double dA = numbers::PI * (R_el_1 + dR_el_1) * (R_el_1 + dR_el_1) - numbers::PI * (R_el_1 - dR_el_1) * (R_el_1 - dR_el_1);
				const double G = (compute_potential(d, N_ref, N_R, dR_el_1) - compute_potential(d, N_ref, N_R, -dR_el_1)) / dA;
				G_vect.push_back(make_pair(d, G));
			}

			FILE* printout = fopen(file_name_res.c_str(),"w");
			for(const auto& G : G_vect)
				fprintf(printout, "%- 1.6e %- 1.6e\n", G.first, G.second);
			fclose(printout);
		}
	}


}
