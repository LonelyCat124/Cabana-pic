//Include Cabana Header and Kokkos headers

#include <Cabana_Core.hpp>

#include <Kokkos_Core.hpp>
#include <Kokkos_Random.hpp>

#include <iostream>

// Include the standard C++ headers
#include <cmath>
#include <cstdio>
#include <fstream>
#include <vector>
#include <random>
#include <sys/stat.h>
#include <sys/time.h>
#include "hdf5.h"
#include "mpi.h"

const double c = 299792458.00000000;
const double epsilon0 = 8.854187817620389850536563031710750e-12;

enum FieldNames{ id = 0,
                 weight, 
                 mass,
                 charge,
                 part_pos,
                 part_p,
                 rank
               };

//Setting default Memory/Host/Execution spaces
//using MemorySpace = Kokkos::CudaSpace;
using MemorySpace = Kokkos::HostSpace;
using ExecutionSpace = Kokkos::DefaultExecutionSpace; /*Kokkos::Serial;*/
using DeviceType = Kokkos::Device<Kokkos::DefaultExecutionSpace, MemorySpace/*ExecutionSpace, MemorySpace*/>;
using HostType = Kokkos::Device<Kokkos::Serial, Kokkos::HostSpace>;
using field_type = Kokkos::View<double* , MemorySpace>;
using host_mirror_type = field_type::HostMirror;
//Types used in the tuple (corresponds to the enum before)
using DataTypes = Cabana::MemberTypes<int64_t, double, double,
                                      double, double[3], double[3],
                                      int>;
//Declare vector length of SoAs
const int VectorLength = 16;

struct field{
    double hdt;
    double hdtx;
    double cnx;
    double fac;
    int field_order;
    double fng;
    double cfl;
    double x_grid_min_local;
    double x_grid_max_local;
};

using field_struct_type = Kokkos::View<struct field*, MemorySpace>;
using field_struct_host = field_struct_type::HostMirror;

struct config_type{
    field_struct_type field;
    field_type ex;
    field_type ey;
    field_type ez;
    field_type bx;
    field_type by;
    field_type bz;
    field_type jx;
    field_type jy;
    field_type jz;
};

struct boundary{
   double x_min, x_max;
};

//Periodic
struct field_bc{

    field_type _field;
    int _ng;
    int _nx;

    KOKKOS_INLINE_FUNCTION
    field_bc(field_type field, int nx, int ng): _field(field),
        _ng(ng), _nx(nx){}

    KOKKOS_INLINE_FUNCTION
    void operator()(const int ix) const{
        _field(_ng+_nx+ix) = _field(_ng+ix);
        _field(ix) = _field(_nx+ix);
    }

};

void field_bc_mpi(field_type field, int nx, int ng){
    
    int myrank = 0; MPI_Comm_rank( MPI_COMM_WORLD, &myrank );
    int nranks = 1; MPI_Comm_size( MPI_COMM_WORLD, &nranks );
    int previous_rank = ( myrank == 0 ) ? nranks - 1 : myrank - 1;
    int next_rank = ( myrank == nranks - 1 ) ? 0 : myrank + 1;
    double* bottom = (double*) malloc(sizeof(double) * ng);
    double* top = (double*) malloc(sizeof(double) * ng);

    MPI_Request send_bottom_request, recv_bottom_request;
    MPI_Request send_top_request, recv_top_request;
    MPI_Isend( &(field.data()[0]), ng, MPI_DOUBLE, previous_rank, 1, MPI_COMM_WORLD, &send_bottom_request); //Send bottom
    MPI_Isend( &(field.data()[nx+ng]), ng, MPI_DOUBLE, next_rank, 2, MPI_COMM_WORLD, &send_top_request); //Send top
    MPI_Irecv( top, ng, MPI_DOUBLE, next_rank, 1, MPI_COMM_WORLD, &recv_top_request); //Recv top
    MPI_Irecv( bottom, ng, MPI_DOUBLE, previous_rank, 2, MPI_COMM_WORLD, &recv_bottom_request); //Recv bottom

    // Do the communication
    MPI_Request arrayreq[4];
    arrayreq[0] = send_bottom_request;
    arrayreq[1] = recv_bottom_request;
    arrayreq[2] = send_top_request;
    arrayreq[3] = recv_top_request;
    MPI_Waitall(4, arrayreq, MPI_STATUSES_IGNORE);

    for(int i = 0; i < ng; i++){
        field(ng+nx+i) = top[i];
        field(i) = bottom[i];
    }

    free(bottom);
    free(top);
}

void efield_bcs(field_type ex, field_type ey, field_type ez,
                int nx, int ng){
 field_bc_mpi(ex, nx, ng);
 field_bc_mpi(ey, nx, ng);
 field_bc_mpi(ez, nx, ng);
}

void bfield_bcs(field_type bx, field_type by, field_type bz,
                int nx, int ng, bool mpi_only){
 field_bc_mpi(bx, nx, ng);
 field_bc_mpi(by, nx, ng);
 field_bc_mpi(bz, nx, ng);
}


//void efield_bcs(field_type ex, field_type ey, field_type ez,
//                int nx, int ng){
//    field_bc ex_fbc(ex, nx, ng);
//    field_bc ey_fbc(ey, nx, ng);
//    field_bc ez_fbc(ez, nx, ng);
//
//    auto rp = Kokkos::RangePolicy<>(0, ng);
//    Kokkos::parallel_for("ex_bcs", rp, ex_fbc);
//    Kokkos::parallel_for("ey_bcs", rp, ey_fbc);
//    Kokkos::parallel_for("ez_bcs", rp, ez_fbc);
//    Kokkos::fence();
//}
//
//void bfield_bcs(field_type bx, field_type by, field_type bz,
//                int nx, int ng, bool mpi_only){
//
//    field_bc bx_fbc(bx, nx, ng);
//    field_bc by_fbc(by, nx, ng);
//    field_bc bz_fbc(bz, nx, ng);
//    auto rp = Kokkos::RangePolicy<>(0, ng);
//    Kokkos::parallel_for("bx_bcs", rp, bx_fbc);
//    Kokkos::parallel_for("by_bcs", rp, by_fbc);
//    Kokkos::parallel_for("bz_bcs", rp, bz_fbc);
//    Kokkos::fence();
//}

//Periodic
struct processor_summation_boundaries{

    field_type _field;
    int _ng;
    int _nx;

    KOKKOS_INLINE_FUNCTION
    processor_summation_boundaries(field_type field, int nx, int ng): _field(field),
        _ng(ng), _nx(nx){}

    KOKKOS_INLINE_FUNCTION
    void operator()(const int ix) const{
        _field(_ng+ix) += _field(_ng+_nx+ix);
        _field(_nx+_ng-ix) += _field(_ng-ix);
    }

};

// Periodic
void processor_summation_boundaries_mpi(field_type field, int nx, int ng){

    int myrank = 0; MPI_Comm_rank( MPI_COMM_WORLD, &myrank );
    int nranks = 1; MPI_Comm_size( MPI_COMM_WORLD, &nranks );
    int previous_rank = ( myrank == 0 ) ? nranks - 1 : myrank - 1;
    int next_rank = ( myrank == nranks - 1 ) ? 0 : myrank + 1;
    double* bottom = (double*) malloc(sizeof(double) * ng);
    double* top = (double*) malloc(sizeof(double) * ng);

    MPI_Request send_bottom_request, recv_bottom_request;
    MPI_Request send_top_request, recv_top_request;
    MPI_Isend( &(field.data()[0]), ng, MPI_DOUBLE, previous_rank, 1, MPI_COMM_WORLD, &send_bottom_request); //Send bottom
    MPI_Isend( &(field.data()[nx+ng]), ng, MPI_DOUBLE, next_rank, 2, MPI_COMM_WORLD, &send_top_request); //Send top
    MPI_Irecv( top, ng, MPI_DOUBLE, next_rank, 1, MPI_COMM_WORLD, &recv_top_request); //Recv top
    MPI_Irecv( bottom, ng, MPI_DOUBLE, previous_rank, 2, MPI_COMM_WORLD, &recv_bottom_request); //Recv bottom

    // Do the communication
    MPI_Request arrayreq[4];
    arrayreq[0] = send_bottom_request;
    arrayreq[1] = recv_bottom_request;
    arrayreq[2] = send_top_request;
    arrayreq[3] = recv_top_request;
    MPI_Waitall(4, arrayreq, MPI_STATUSES_IGNORE);

    for(int i = 0; i < ng; i++){
        field(nx+i) += top[i];
        field(nx+ng-1-i) += bottom[ng-1-i];
    }
    free(bottom);
    free(top);
    
}


void current_bcs(field_type jx, field_type jy, field_type jz,
                 int nx, int ng){
processor_summation_boundaries_mpi(jx, nx, ng);
processor_summation_boundaries_mpi(jy, nx, ng);
processor_summation_boundaries_mpi(jz, nx, ng);
}



//void current_bcs(field_type jx, field_type jy, field_type jz,
//                 int nx, int ng){
//
//    processor_summation_boundaries jx_sum(jx, nx, ng);
//    processor_summation_boundaries jy_sum(jy, nx, ng);
//    processor_summation_boundaries jz_sum(jz, nx, ng);
//    auto rp = Kokkos::RangePolicy<>(0, ng);
//    Kokkos::parallel_for("jx_summation_boundaries", rp, jx_sum);
//    Kokkos::parallel_for("jy_summation_boundaries", rp, jy_sum);
//    Kokkos::parallel_for("jz_summation_boundaries", rp, jz_sum);
//    Kokkos::fence();
//}


void current_finish(field_type jx, field_type jy, field_type jz,
                    int nx, int ng){
  current_bcs(jx, jy, jz, nx, ng);
  field_bc_mpi(jx, nx, ng);
  field_bc_mpi(jy, nx, ng);
  field_bc_mpi(jz, nx, ng);
}


//void current_finish(field_type jx, field_type jy, field_type jz,
//                 int nx, int ng){
//
//    current_bcs(jx, jy, jz, nx, ng);
//
//    field_bc jx_fbc(jx, nx, ng);
//    field_bc jy_fbc(jy, nx, ng);
//    field_bc jz_fbc(jz, nx, ng);
//    auto rp = Kokkos::RangePolicy<>(0, ng);
//    Kokkos::parallel_for("jx_bcs", rp, jx_fbc);
//    Kokkos::parallel_for("jy_bcs", rp, jy_fbc);
//    Kokkos::parallel_for("jz_bcs", rp, jz_fbc);
//    Kokkos::fence();
//}

void bfield_final_bcs(field_type bx, field_type by, field_type bz, int nx, int ng){
    //Ignore update_laser_omegs
    bfield_bcs(bx, by, bz, nx, ng, true);
    //x_min_boundary ignore
    //x_max_boundary ignore
    //
    bfield_bcs(bx, by, bz, nx, ng, false);
}

struct update_e_field_functor{

    //TODO
    config_type _config;
//    int _nx;

    KOKKOS_INLINE_FUNCTION
    update_e_field_functor(config_type config/*, int nx*/) : _config(config)/*,*/
                           /*_nx(nx)*/ {}

    KOKKOS_INLINE_FUNCTION
    void operator()(const int ix) const{
        double cpml_x;
        double c1, c2, c3;
        double cx1, cx2, cx3;
    
        //Assuming non cpml
        if(_config.field(0).field_order == 2){
           cx1 = _config.field(0).cnx;
           _config.ex(ix) = _config.ex(ix) - _config.field(0).fac * _config.jx(ix);
           int m1 = ix-1;
           _config.ey(ix) = _config.ey(ix) - cx1 * (_config.bz(ix) - _config.bz(m1)) - _config.field(0).fac * _config.jy(ix);
           _config.ez(ix) = _config.ez(ix) + cx1 * (_config.by(ix) - _config.by(m1)) - _config.field(0).fac * _config.jz(ix);
        }else if(_config.field(0).field_order == 4){
            c1 = 9.0/8.0;
            c2 = -1.0 / 24.0;
            cx1 = c1 * _config.field(0).cnx;
            cx2 = c2 * _config.field(0).cnx;
            _config.ex(ix) = _config.ex(ix) - _config.field(0).fac * _config.jx(ix);
            int m1 = ix-1;
            int m2 = ix-2;
            int p1 = ix+1;
            _config.ey(ix) = _config.ey(ix) - cx1 * (_config.bz(ix) - _config.bz(m1)) - cx2 *
                             (_config.bz(p1) - _config.bz(m2)) - _config.field(0).fac * _config.jy(ix);
            _config.ez(ix) = _config.ez(ix) + cx1 * (_config.by(ix) - _config.by(m1)) + cx2 *
                             (_config.by(p1) - _config.by(m2)) - _config.field(0).fac * _config.jz(ix);
        }else{
            c1 = 75.0/64.0;
            c2 = -25.0 / 384.0;
            c3 = 3.0 / 640.0;
            cx1 = c1 * _config.field(0).cnx;
            cx2 = c2 * _config.field(0).cnx;
            cx3 = c3 * _config.field(0).cnx;

            int m1 = ix-1;
            int m2 = ix-2;
            int m3 = ix-3;
            int p1 = ix+1;
            int p2 = ix+2;
            _config.ex(ix) = _config.ex(ix) - _config.field(0).fac * _config.jx(ix);
            _config.ey(ix) = _config.ey(ix) - cx1 * (_config.bz(ix) - _config.bz(m1))
                   - cx2 * (_config.bz(p1) - _config.bz(m2)) - cx3 * (_config.bz(p2) - _config.bz(m3)) -
                   _config.field(0).fac * _config.jy(ix);
            _config.ez(ix) = _config.ez(ix) + cx1 * (_config.by(ix) - _config.by(m1))
                   + cx1 * (_config.by(p1) - _config.by(m2)) + cx3 * (_config.by(p2) - _config.by(m3)) - 
                   _config.field(0).fac * _config.jy(ix);
        }
    }
};

struct update_b_field_functor{
    //TODO
    config_type _config;
    /*field_type _ex, _ey, _ez;
    field_type _bx, _by, _bz;
    field_struct_type _field;*/
/*    struct field _field;*/
//    int _nx;

    KOKKOS_INLINE_FUNCTION
    update_b_field_functor(config_type config/*, int nx*/) : _config(config)/*, _nx(nx)*/ {}

/*    KOKKOS_INLINE_FUNCTION
    void update_field(struct field &field){
        _field.hdt = field.hdt;
        _field.hdtx = field.hdtx;
        _field.cnx = field.cnx;
        _field.fac = field.fac;
    }*/

    KOKKOS_INLINE_FUNCTION
    void operator()(const int ix) const{
        double cpml_x;
        double c1, c2, c3;
        double cx1, cx2, cx3;
        //Assuming non cpml and using maxwell_solver_yee
        if(_config.field(0).field_order == 2){
            //Yee solver
            cx1 = _config.field(0).hdtx;
            int p1 = ix+1;
            _config.by(ix) = _config.by(ix) + cx1 * (_config.ez(p1) - _config.ez(ix));
        }else if(_config.field(0).field_order == 4){
            c1 = 9.0/8.0;
            c2 = -1.0 / 24.0;
    
            cx1 = c1 * _config.field(0).hdtx;
            cx2 = c2 * _config.field(0).hdtx;

            int m1 = ix-1;
            int p1 = ix+1;
            int p2 = ix+2;
            _config.by(ix) = _config.by(ix) + cx1 * (_config.ez(p1) - _config.ez(ix)) + cx2 * (_config.ez(p2) - _config.ez(m1));
            _config.bz(ix) = _config.bz(ix) - cx1 * (_config.ey(p1) - _config.ey(ix)) - cx2 * (_config.ey(p2) - _config.ey(m1));

        }else{
            c1 = 75.0/64.0;
            c2 = -25.0 / 384.0;
            c3 = 3.0 / 640.0;

            cx1 = c1 * _config.field(0).hdtx;
            cx2 = c2 * _config.field(0).hdtx;
            cx3 = c3 * _config.field(0).hdtx;

            int m1 = ix-1;
            int m2 = ix-2;
            int p1 = ix+1;
            int p2 = ix+2;
            int p3 = ix+3;
            _config.by(ix) = _config.by(ix) + cx1 * (_config.ez(p1) - _config.ez(ix)) + cx2 * (_config.ez(p2) - _config.ez(m1)) +
                              cx3 * (_config.ez(p3) - _config.ez(m2));
            _config.bz(ix) = _config.bz(ix) - cx1 * (_config.ey(p1) - _config.ey(ix)) - cx2 * (_config.ey(p2) - _config.ey(m1)) -
                              cx3 * (_config.ey(p3) - _config.ey(m2));
        }
    }
};

KOKKOS_INLINE_FUNCTION
void interpolate_from_grid_tophat(double part_weight, double part_q, double part_m,
                                  double *part_x, double part_px, double part_py, double part_pz,
                                  double *ex_part, double *ey_part, double *ez_part,
                                  double *bx_part, double *by_part, double *bz_part,
                                  double cell_x_r, int *cell_x1, double *gx, double *hx,
                                  field_type ex, field_type ey, field_type ez,
                                  field_type bx, field_type by, field_type bz, double idt,
                                  double idx, double dtco2, double idtf, double idxf,
                                  int nx, double fcx, double fcy, int ng){

    double part_mc = c * part_m;
    double ipart_mc = 1.0 / part_mc;

    double part_ux = part_px * ipart_mc;
    double part_uy = part_py * ipart_mc;
    double part_uz = part_pz * ipart_mc;

    *cell_x1 = floor(cell_x_r + 0.5);
    double cell_frac_x = ((double)(*cell_x1)) - cell_x_r;
    *cell_x1 = (*cell_x1) + 1;

    //First tophat stuff, gx.inc
    gx[0] = 0.5 + cell_frac_x;
    gx[1] = 0.5 - cell_frac_x;

    int cell_x2 = floor(cell_x_r);
    cell_frac_x = ((double)(cell_x2)) - cell_x_r + 0.5;
    cell_x2 = cell_x2+1;

    int dcellx = 0;
    hx[dcellx] = 0.5 + cell_frac_x;
    hx[dcellx+1] = 0.5 - cell_frac_x;

    //Shift all index accesses up by ng compared to the FDPS version
    *ex_part = hx[0] * ex(cell_x2 + ng) + hx[1] * ex(cell_x2 + 1 + ng);
    *ey_part = gx[0] * ey((*cell_x1) + ng) + gx[1] * ey( (*cell_x1) + 1 + ng);
    *ez_part = gx[0] * ez((*cell_x1) + ng) + gx[1] * ez( (*cell_x1) + 1 + ng);

    *bx_part = gx[0] * bx((*cell_x1) + ng) + gx[1] * bx((*cell_x1) + 1 + ng);
    *by_part = hx[0] * by(cell_x2 + ng) + hx[1] * by(cell_x2 + 1 + ng);
/*    if (std::isinf(*by_part)){
        std::cout << "by_part is inf" << hx[0] << ", " << cell_x2+ng << ", " << by(cell_x2 + 1 + ng) << "\n";
        abort();
    }*/
    *bz_part = hx[0] * bz(cell_x2 + ng) + hx[1] * bz(cell_x2 + 1 + ng);
//    std::cout << "vals: " << *bz_part << ", " << dcellx << ", " << cell_x2 + ng << ", " << bz(cell_x2 + ng) << "\n";
//    std::cout << "cont: " << hx[0] << ", " << hx[1] << ", " << cell_x2 + 1 + ng << ", " << bz(cell_x2 + 1 + ng) << ", " << bz.size() << "\n";

}

KOKKOS_INLINE_FUNCTION
void GatherForcesToGrid(double part_weight, double part_q,
                        double part_x, double delta_x,
                        int cell_x1, double *gx, double *hx,
                        field_type jx, field_type jy, field_type jz,
                        double idt, double part_vy, double part_vz,
                        double idx, double dtco2, double idtf, double idxf,
                        int nx, int id, double fcx, double fcy, int jng){

    //Move particle to t + 1.5dt;
    part_x = part_x + delta_x;
    double cell_x_r = part_x * idx - 0.5;
    int cell_x3 = floor(cell_x_r + 0.5);
    double cell_frac_x = ((double)(cell_x3)) - cell_x_r;
    cell_x3 = cell_x3 + 1;

    for(int i = -1; i < 3; i++){
        hx[i] = 0.0;
    }

    int dcellx = cell_x3 - cell_x1;//Centered on 0 like epoch
    hx[dcellx] = 0.5 + cell_frac_x;
    hx[dcellx+1] = 0.5 - cell_frac_x;

    for(int i = -1; i < 3; i++){
        hx[i] = hx[i] - gx[i];
    }

    int xmin = 0 + (dcellx -1)/ 2;
    int xmax = 1 + (dcellx + 1)/ 2;

    double fjx = fcx * part_q;
    double fjy = fcy * part_q * part_vy;
    double fjz = fcy * part_q * part_vz;

    double jxh = 0.0;

    for(int ix = xmin; ix <= xmax; ix++){
        int cx = cell_x1 + ix;

        double wx = hx[ix];
        double wy = gx[ix] + 0.5 * hx[ix];

        //This is the bit that actually solves d(rho)/dt = -div(J)
        jxh = jxh - fjx * wx;

        double jyh = fjy * wy;
        double jzh = fjz * wy;
        //Scatterview could be used here
        Kokkos::atomic_add_fetch(&(jx(cx+jng)), jxh);
        Kokkos::atomic_add_fetch(&(jy(cx+jng)), jyh);
        Kokkos::atomic_add_fetch(&(jz(cx+jng)), jzh);
    }
}

template<class IDSlice, class WeightSlice,
         class MassSlice, class ChargeSlice, 
         class PartPosSlice, class PartPSlice>
struct push_particles_functor{

    double _dt;
    double _dx;
    config_type _config;
/*    field_type _jx;
    field_type _jy;
    field_type _jz;
    field_type _ex;
    field_type _ey;
    field_type _ez;
    field_type _bx;
    field_type _by;
    field_type _bz;
    field_struct_type _field;*/
    int _nx;
    int _ng;
    int _jng;


    IDSlice _id;
    WeightSlice _weight;
    MassSlice _mass;
    ChargeSlice _charge;
    PartPosSlice _part_pos;
    PartPSlice _part_p;
    KOKKOS_INLINE_FUNCTION
    push_particles_functor( IDSlice id, WeightSlice weight, MassSlice mass,
           ChargeSlice charge, PartPosSlice part_pos, PartPSlice part_p,
           double dt, double dx, config_type config, int nx, int ng, int jng) : _dt(dt), _dx(dx), _config(config),
           _nx(nx), _ng(ng), _jng(jng), _id(id), _weight(weight),
           _mass(mass), _charge(charge), _part_pos(part_pos), _part_p(part_p){
    }

/*    KOKKOS_INLINE_FUNCTION
    void update_field(struct field &field){
        _field.hdt = field.hdt;
        _field.hdtx = field.hdtx;
        _field.cnx = field.cnx;
        _field.fac = field.fac;
    }*/

    KOKKOS_INLINE_FUNCTION
    void operator()( const int i, const int a ) const{
        double idt = 1.0 / _dt;
        double idx = 1.0 / _dx;
        double dto2 = _dt / 2.0;
        double dtco2 = c * dto2;
        double dtfac = 0.5 * _dt;

        double idtf = idt;
        double idxf = idx;

        double gxarray[4] =  {0.0, 0.0, 0., 0.};
        double hxarray[4] =  {0.0, 0.0, 0., 0.};

        double* gx = &gxarray[1];
        double* hx = &hxarray[1];


        double part_weight = _weight.access(i, a);
        double fcx = idtf * part_weight;
        double fcy = idxf * part_weight;

        double part_q = _charge.access(i, a);
        double part_m = _mass.access(i, a);
        double part_mc = c * part_m;
        double ipart_mc = 1.0 / part_mc;
        double cmratio = part_q * dtfac * ipart_mc;
        double ccmratio = c * cmratio;

        double part_x = _part_pos.access(i,a,0) - _config.field(0).x_grid_min_local;
        double part_ux = _part_p.access(i,a,0) * ipart_mc;
        double part_uy = _part_p.access(i,a,1) * ipart_mc;
        double part_uz = _part_p.access(i,a,2) * ipart_mc;


        //Calculate v(t) from p(t)
        double gamma_rel = sqrtf(part_ux*part_ux + part_uy*part_uy + part_uz*part_uz + 1.0);
        double root = dtco2 / gamma_rel;

        //Move particles to half timestep position to first order
        part_x = part_x + part_ux * root;

        double cell_x_r = part_x * idx - 0.5;

        double ex_part = 0.0;
        double ey_part = 0.0;
        double ez_part = 0.0;
        double bx_part = 0.0;
        double by_part = 0.0;
        double bz_part = 0.0;
        int cell_x1;
        
        interpolate_from_grid_tophat(part_weight, part_q, part_m,
                                     &part_x, _part_p.access(i,a,0), _part_p.access(i,a,1), _part_p.access(i,a,2),
                                     &ex_part, &ey_part, &ez_part,
                                     &bx_part, &by_part, &bz_part,
                                     cell_x_r, &cell_x1, gx, hx,
                                     _config.ex, _config.ey, _config.ez, _config.bx, _config.by, _config.bz,
                                     idt, idx, dtco2, idtf, idxf,
                                     _nx, fcx, fcy, _ng);

        double uxm = part_ux + cmratio * ex_part;
        double uym = part_uy + cmratio * ey_part;
        double uzm = part_uz + cmratio * ez_part;

        //Half timestep, use Boris 1970 rotation
        gamma_rel = sqrt(uxm*uxm + uym*uym + uzm*uzm + 1.0);
        root = ccmratio/gamma_rel;

        double taux = bx_part * root;
        double tauy = by_part * root;
        double tauz = bz_part * root;

        double taux2 = taux * taux;
        double tauy2 = tauy * tauy;
        double tauz2 = tauz * tauz;

        double tau = 1.0 / (1.0 + taux2 + tauy2 + tauz2);

        double uxp = ((1.0 + taux2 - tauy2 - tauz2) * uxm
            + 2.0 * ((taux * tauy + tauz) * uym
            + (taux * tauz - tauy) * uzm)) * tau;
        double uyp = ((1.0 - taux2 + tauy2 - tauz2) * uym
            + 2.0 * ((tauy * tauz + taux) * uzm
            + (tauy * taux - tauz) * uxm)) * tau;
        double uzp = ((1.0 - taux2 - tauy2 + tauz2) * uzm
            + 2.0 * ((tauz * taux + tauy) * uxm
            + (tauz * tauy - taux) * uym)) * tau;

        part_ux = uxp + cmratio * ex_part;
        part_uy = uyp + cmratio * ey_part;
        part_uz = uzp + cmratio * ez_part;

        double part_u2 = part_ux*part_ux + part_uy*part_uy + part_uz*part_uz;
        gamma_rel = sqrt(part_u2 + 1.0);
        root = c / gamma_rel;

        double delta_x = part_ux * root * dto2;
        double part_vy = part_uy * root;
        double part_vz = part_uz * root;

        part_x = part_x + delta_x;

        _part_pos.access(i,a,0) = part_x + _config.field(0).x_grid_min_local;
        _part_p.access(i,a,0) = part_mc * part_ux;
        _part_p.access(i,a,1) = part_mc * part_uy;
        _part_p.access(i,a,2) = part_mc * part_uz;

        //TODO GatherForcesToGrid
        GatherForcesToGrid(part_weight, part_q, part_x, delta_x,
                        cell_x1, gx, hx, _config.jx, _config.jy, _config.jz,
                        idt, part_vy, part_vz, idx, dtco2,
                        idtf, idxf, _nx, id, fcx, fcy, _jng);
    }
};


void update_eb_fields_half(field_type &ex, field_type &ey, field_type &ez, int nx,
                           field_type &jx, field_type &jy, field_type &jz,
                           field_type &bx, field_type &by, field_type &bz,
                           double dt, double dx, field_struct_host host_field,
                           field_struct_type field, int ng,
                           update_e_field_functor &update_e_field,
                           update_b_field_functor &update_b_field,
                           Kokkos::RangePolicy<> rangepolicy){

    host_field(0).hdt = 0.5 * dt;
    host_field(0).hdtx = host_field(0).hdt / dx;
    host_field(0).cnx = host_field(0).hdtx * c*c;
    host_field(0).fac = host_field(0).hdt / epsilon0;
    Kokkos::deep_copy(field, host_field);

//    update_e_field.update_field(field);

    Kokkos::parallel_for("update_e_field", rangepolicy, update_e_field);
    Kokkos::fence();

    efield_bcs(ex, ey, ez, nx, ng);

//    update_b_field.update_field(field);
    Kokkos::parallel_for("update_b_field", rangepolicy, update_b_field);
    Kokkos::fence();

    bfield_bcs(bx, by, bz, nx, ng, true);

}

void update_eb_fields_final(field_type &ex, field_type &ey, field_type &ez, int nx,
                            field_type &jx, field_type &jy, field_type &jz,
                            field_type &bx, field_type &by, field_type &bz,
                            double dt, double dx, field_struct_host host_field, field_struct_type field, int ng,
                            update_e_field_functor &update_e_field,
                            update_b_field_functor &update_b_field,
                            Kokkos::RangePolicy<> rangepolicy){

    host_field(0).hdt = 0.5 * dt;
    host_field(0).hdtx = host_field(0).hdt / dx;
    host_field(0).cnx = host_field(0).hdtx * c*c;
    host_field(0).fac = host_field(0).hdt / epsilon0;

    Kokkos::deep_copy(field, host_field);
//    update_b_field.update_field(field);
    Kokkos::parallel_for("update_b_field", rangepolicy, update_b_field);
    Kokkos::fence();

    bfield_final_bcs(bx, by, bz, nx, ng);

//    update_e_field.update_field(field);
    Kokkos::parallel_for("update_e_field", rangepolicy, update_e_field);
    Kokkos::fence();

    efield_bcs(ex, ey, ez, nx, ng);
}

//TODO Particle Push

//TODO Particle bcs
template<class PartPosSlice, class RankSlice>
struct particle_bcs_functor{
    boundary _box;
    PartPosSlice _part_pos;
    RankSlice _rank;
    int _nranks;
    //TODO The slice?

    KOKKOS_INLINE_FUNCTION
    particle_bcs_functor(boundary box, PartPosSlice part_poss, RankSlice rank):
        _box(box), _part_pos(part_poss), _rank(rank){
       _nranks = 1; 
        MPI_Comm_size( MPI_COMM_WORLD, &_nranks );
    }

    //SIMD Parallelised implementation for now
    KOKKOS_INLINE_FUNCTION
    void operator()(const int ix, const int ij) const{
        if (_part_pos.access(ix, ij,0) >= _box.x_max){
            _part_pos.access(ix, ij,0) -= (_box.x_max - _box.x_min);
        }
        if (_part_pos.access(ix, ij,0) < _box.x_min){
            _part_pos.access(ix, ij,0) += (_box.x_max - _box.x_min);
        }

        //Compute rank for this particle to go to.
        double size = _box.x_max - _box.x_min;
        // Compute relative position to the minimum
        double position = _part_pos.access(ix, ij,0) - _box.x_min;
        double size_per_rank = size / ((double) _nranks);
        int rank = (int)(position/size_per_rank);
        _rank.access(ix, ij) = rank;
    }

};


std::default_random_engine generator;
double momentum_from_temperature(double mass, double temperature, double drift){
  double stdev, mu;
  const double kb = 1.3806488e-23;  // J/K
  stdev = sqrt(temperature * mass * kb);
  mu = drift;
  //random_box_muller
  std::normal_distribution<double> d{mu, stdev};
  double x = d(generator);
  return x;
}

//TODO Initialisation
//
//

void set_field_order(int order, field_struct_host &field /* struct field *field*/){

    field(0).field_order = order;
    field(0).fng = (double)(field(0).field_order) / 2.0;
    if (field(0).field_order == 2){
       field(0).cfl = 1.0;
    }else if(field(0).field_order == 4){
        field(0).cfl = 6.0/7.0;
    }else{
        field(0).cfl = 120.0/149.0;
    }
}

void minimal_init(field_struct_host &field /*struct field *field*/, double x_grid_min_local,
                   double x_grid_max_local){

    //Real is double precision
    //
    //dt_plasma_frequency = 0.0
    //dt_multiplier = 0.95
    //stdout_frequency = 0
    //cpml_thickness = 6
    //cpml_kappa_max = 20.0
    //cpml_a_max = 0.15
    //cpml_sigma_max = 0.7
    //cpml_x_min_offset = 0
    //cpml_x_max_offset = 0

    //npart_global = -1
    //smooth_currents = .FALSE.
    //use_balance = .FALSE.
    //use_random_seed = .FALSE.
    //use_offset_grid = .FALSE.
    //use_particle_lists = .FALSE.
    //use_multiphoton = .TRUE.
    //use_bsi = .TRUE.
    //need_random_state = .FALSE.
    //force_first_to_be_restartable = .FALSE.
    //force_final_to_be_restartable = .FALSE.
    //full_dump_every = -1
    //restart_dump_every = -1
    //nsteps = -1
    //t_end = HUGE(1.0)
    //particles_max_id = 0
    //n_zeros = 4

    //laser_inject_local = 0.0
    //laser_absorb_local = 0.0
    //old_elapsed_time = 0.0
    //window_offset = 0.0
    //
    set_field_order(2, field);
    field(0).x_grid_min_local = x_grid_min_local;
    field(0).x_grid_max_local = x_grid_max_local;
  //  field(0).x_grid_min_local = 3.5087719298245611E-005;
//    field(0).x_grid_max_local = 1.9999649122807017;
/*    field->x_grid_min_local = 3.5087719298245611E-005;
    field->x_grid_max_local = 1.9999649122807017;*/
    //eval stack stuff
}


struct init_eb_arrays_functor{
    host_mirror_type _ex, _ey, _ez;
    host_mirror_type _bx, _by, _bz;

    KOKKOS_INLINE_FUNCTION
    init_eb_arrays_functor(host_mirror_type ex, host_mirror_type ey, host_mirror_type ez,
                           host_mirror_type bx, host_mirror_type by, host_mirror_type bz) :
                           _ex(ex), _ey(ey), _ez(ez), _bx(bx),
                            _by(by), _bz(bz) {}

    KOKKOS_INLINE_FUNCTION
    void operator()(const int ix) const{
        _ex(ix) = 0.0;
        _ey(ix) = 0.0;
        _ez(ix) = 0.0;
        _bx(ix) = 0.0;
        _by(ix) = 0.0;
        _bz(ix) = 2.1;
    }
};

struct init_j_arrays_functor{
    host_mirror_type _jx, _jy, _jz;

    KOKKOS_INLINE_FUNCTION
    init_j_arrays_functor(host_mirror_type jx, host_mirror_type jy, host_mirror_type jz) :
	                  _jx(jx), _jy(jy), _jz(jz) {}

    KOKKOS_INLINE_FUNCTION
    void operator()(const int ix) const{
        _jx(ix) = 0.0;
        _jy(ix) = 0.0;
        _jz(ix) = 0.0;
    }
};	

void set_initial_values(host_mirror_type &ex,
                        host_mirror_type &ey,
                        host_mirror_type &ez,
                        host_mirror_type &bx,
                        host_mirror_type &by,
                        host_mirror_type &bz,
                        host_mirror_type &jx,
                        host_mirror_type &jy,
                        host_mirror_type &jz,
                        int nx, int ng, int jng){

    init_eb_arrays_functor init_eb(ex, ey, ez, bx, by, bz);
    init_j_arrays_functor init_j(jx, jy, jz);
    Kokkos::parallel_for("Init e&b arrays", Kokkos::RangePolicy<Kokkos::OpenMP>(0,nx+2*ng), 
            init_eb);
    Kokkos::parallel_for("Init j arrays", Kokkos::RangePolicy<Kokkos::OpenMP>(0,nx + 2*jng),
            init_j);
    Kokkos::fence(); 
   srand(7842432); 
}
void after_control(field_type::HostMirror &ex,
                   field_type::HostMirror &ey,
                   field_type::HostMirror &ez,
                   field_type::HostMirror &bx,
                   field_type::HostMirror &by,
                   field_type::HostMirror &bz,
                   field_type::HostMirror &jx,
                   field_type::HostMirror &jy,
                   field_type::HostMirror &jz,
                   int nx, int ng, int jng){

    //setup_grid (ignore for single node)
    set_initial_values(ex, ey, ez, bx, by, bz, jx, jy, jz, nx, ng, jng);
}

void auto_load(Cabana::AoSoA<DataTypes, HostType, VectorLength> &particle_aosoa,
               boundary box, int npart, int ncell, double cell_size, int nxglobal,
               double dx, int myrank, int nranks, int mincell, int maxcell){

    // Still not using ghost cells yet, just allreducing the current array.
    int i0 = 1; //-ng, number ghost cells, which we're setting to 0
    int i1 = 0; // 1 - i0

    //if pre_loading - not sure what this is for.
    int *npart_per_cell_array = (int*)malloc(sizeof(int) * 2); //i0:nx+i1 in fortran
    for (int i = 0; i < 2; i++){
        npart_per_cell_array[i] = 0;
    }

    //For each specie
    //Grab species initial condition
    double species_charge[2];
    species_charge[0] = -1.0;
    species_charge[1] = 1.0;
    double species_mass[2];
    species_mass[0] = 1.0;
    species_mass[1] = 3670.48294;

    double species_density[2];
    species_density[0] = 1e19;
    species_density[1] = (1.0 - 2.0 * 1e-3) * 1e19;

    double *density = (double*)malloc(sizeof(double) * nxglobal);
    bool *density_map = (bool*)malloc(sizeof(bool) * nxglobal);

    double species_count[2];
    species_count[0] = npart/2;
    species_count[1] = npart/2;
    if(npart % 2 != 0){
        if(myrank % 2 == 0){
            species_count[0]++;
        }else{
            species_count[1]++;
        }
    }
    double counter = 0;

    auto id_slice = Cabana::slice<id>(particle_aosoa);
    auto mass_slice = Cabana::slice<mass>(particle_aosoa);
    auto charge_slice = Cabana::slice<charge>(particle_aosoa);
    auto weight_slice = Cabana::slice<weight>(particle_aosoa);
    auto part_pos_slice = Cabana::slice<part_pos>(particle_aosoa);
    auto part_p_slice = Cabana::slice<part_p>(particle_aosoa);
    for(int i = 0; i < 2; i++){
        double npart_per_cell_average = (double)(species_count[i]) / (double)(maxcell-mincell);
        if(npart_per_cell_average <= 0) continue;
        for(int ix = mincell; ix < maxcell; ix++){
            density[ix] = species_density[i];
            density_map[ix] = true;
        }
        int ipart = 0;
        int start = counter;
        for(int ix = mincell;  ix < maxcell; ix++){
            int npart_per_cell = round(npart_per_cell_average); //Assuming density for every cell is the same for now
            for(int ipart = 0; ipart < npart_per_cell; ipart++){
                if (counter > particle_aosoa.size()) break;
                id_slice(counter) = counter + (myrank * particle_aosoa.size());
                charge_slice(counter) = species_charge[i] * 1.602176565e-19;
                mass_slice(counter) = species_mass[i] * 9.10938291e-31;
                double x = (double)(ix+1) * cell_size;
                double r = (double) ((double)rand() / ((double)RAND_MAX));
                part_pos_slice(counter, 0) = x + (r - 0.5) * cell_size;
                counter++;

            }
        }
        //TODO From Line 888 FIXME
        //Load particles finished
        int *npart_in_cell = (int*)malloc(sizeof(int) * nxglobal);
        for(int ii = mincell; ii < maxcell; ii++){
            npart_in_cell[ii] = 0;
        }
        // Get the global densities. Its possible this can be optimised but this is easy TODO
        MPI_Allreduce(MPI_IN_PLACE, density, nxglobal, MPI_INT, MPI_SUM, MPI_COMM_WORLD);
        //MPI_Allreduce(MPI_IN_PLACE, density_map, nxglobal, MPI_CXX_BOOL, MPI_SUM, MPI_COMM_WORLD); //Always true so ignored FIXME
        //Loop over particles set
        for(int ipart = start; ipart < counter; ipart++){
            double cell_x_r = part_pos_slice(ipart, 0) / dx - 0.5;
            int cell_x = floor(cell_x_r + 0.5);
            double cell_frac_x = (double)(cell_x) - cell_x_r;
            cell_x = cell_x + 1;
            double gx[2];
            gx[0] = 0.5 + cell_frac_x;
            gx[1] = 0.5 - cell_frac_x;
            //      Calculate density at the particle position
            double wdata = 0.0;
            for(int isubx = 0; isubx < 2; isubx++){
                int ii = cell_x + isubx;
                if (ii >= nxglobal) ii -= nxglobal;
                //if (!density_map[ii]) ii = cell_x + 1 - isubx; FIXME For now density_map is always true so this never happens.
                wdata = wdata + gx[isubx] * density[ii];
            }
            weight_slice(ipart) = wdata;
            if( gx[1] > gx[0]) cell_x = cell_x + 1;
            if (cell_x >= nxglobal) cell_x -= nxglobal;
            npart_in_cell[cell_x] = npart_in_cell[cell_x] + 1;
        }
        //Loop again to normalise the weights
        double wdata = dx;
        for(int ipart = start; ipart < counter; ipart++){
            int cell_x = floor(part_pos_slice( ipart, 0) / dx + 1.5);
            if(cell_x >= nxglobal) cell_x -= nxglobal;
            double lweight = weight_slice(ipart);
            weight_slice(ipart) = lweight * wdata / npart_in_cell[cell_x];
        }
        free(npart_in_cell);
        std::cout << counter << " particles loaded\n";
        auto rank_slice = Cabana::slice<rank>(particle_aosoa);
        particle_bcs_functor<decltype(part_pos_slice), decltype(rank_slice)> pbf(box, part_pos_slice, rank_slice);
        //Cabana::SimdPolicy<VectorLength, ExecutionSpace> simd_policy( 0,
//                                              particle_aosoa.size());
        Cabana::SimdPolicy<VectorLength, Kokkos::OpenMP> simd_policy( 0,
                                              particle_aosoa.size());
        Cabana::simd_parallel_for( simd_policy, pbf, "init_boundary_cond");
        Kokkos::fence();
        //
        double **species_temp = (double**) malloc(sizeof(double*) * 3);
        species_temp[0] = (double*) malloc(sizeof(double) * nxglobal);
        species_temp[1] = (double*) malloc(sizeof(double) * nxglobal);
        species_temp[2] = (double*) malloc(sizeof(double) * nxglobal);
        for(int n = 0; n < 3; n++){
            for(int ix = 0; ix < nxglobal; ix++){
                //We use a specific temperature everywhere for this example
                species_temp[n][ix] = 11594200.000;
            }
        }
        //TODO Fix species_temp in the future.
        //setup_ic_drift
        //Drift is all 0 for now.
        for(int n = 0; n < 3; n++){
            for(int ipart = start; ipart < counter; ipart++){
                double lmass = mass_slice(ipart);
                //include particle_to_grid.inc
                double cell_x_r = (part_pos_slice(ipart, 0)) / dx - 0.5;
                int cell_x = floor(cell_x_r + 0.5);
                double cell_frac_x = (double)(cell_x) - cell_x_r;
                double gx[2];
                gx[0] = 0.5 + cell_frac_x;
                gx[1] = 0.5 - cell_frac_x;
                double temp_local = 0.0;
                double drift_local = 0.0;
                for(int ix = 0; ix < 2; ix++){
                    int index = cell_x + ix;
                    if (index >= nxglobal) index -= nxglobal; //All values are the same so doesn't matter
                    temp_local += gx[ix] * 11594200.000; //FIXME Species temp is fixed for now, species_temp[n][index];
                    drift_local += gx[ix] * 0.0; // Drift is 0
                }
                //Calculate momentum, directions are 0/1/2 to x/y/z
                part_p_slice(ipart, n) = momentum_from_temperature(lmass, temp_local, drift_local);
            }
        }
        free(species_temp[0]);
        free(species_temp[1]);
        free(species_temp[2]);
        free(species_temp);
    }

    free(npart_per_cell_array);
    free(density);
    free(density_map);
}

struct current_start_functor{

    field_type _jx;
    field_type _jy;
    field_type _jz;

    KOKKOS_INLINE_FUNCTION
    current_start_functor(field_type jx, field_type jy, field_type jz) :
        _jx(jx), _jy(jy), _jz(jz) {}

    KOKKOS_INLINE_FUNCTION
    void operator()(const int ix) const{
        _jx(ix) = 0.0;
        _jy(ix) = 0.0;
        _jz(ix) = 0.0;
    }
};


void current_start(field_type jx, field_type jy, field_type jz, int nxlocal, int jng){
    current_start_functor current_start(jx, jy, jz);
    Kokkos::parallel_for("Current start", nxlocal + 2*jng, current_start);
    Kokkos::fence();
}
//void current_start(field_type jx, field_type jy, field_type jz, int nxglobal, int jng){
//
//        current_start_functor current_start(jx, jy, jz);
//        Kokkos::parallel_for("Current start", nxglobal + 2*jng,
//                current_start);
//
//        Kokkos::fence();
//}

void output_routines(Cabana::AoSoA<DataTypes, HostType, VectorLength> particle_aosoa,
                     host_mirror_type ex, host_mirror_type ey, host_mirror_type ez,
                     host_mirror_type bx, host_mirror_type by, host_mirror_type bz, 
		     host_mirror_type jx, host_mirror_type jy, host_mirror_type jz,
                     int output, int grid_size, int ng, int jng){

    // Create HDF5 file
    char filename[250];
    int myrank; MPI_Comm_rank( MPI_COMM_WORLD, &myrank );
    sprintf(filename, "%.4d_%.4d.hdf5", output, myrank);
    hid_t file_id = H5Fcreate(filename, H5F_ACC_TRUNC, H5P_DEFAULT, H5P_DEFAULT);
    hsize_t h_dims[1];
    h_dims[0] = particle_aosoa.size();
    hid_t dim = H5Screate_simple(1, h_dims, NULL);

    //Create a temporary array to store particle data
    double *temp = (double*)malloc(sizeof(double) * particle_aosoa.size());
    hid_t positions = H5Dcreate2(file_id, "position", H5T_NATIVE_DOUBLE, dim, H5P_DEFAULT, H5P_DEFAULT, H5P_DEFAULT);
    //Store the positions
    for(int i = 0; i < particle_aosoa.size(); i++){
        auto part = particle_aosoa.getTuple(i);
        double pos = Cabana::get<part_pos>(part,0);
        temp[i] = pos;
    }
    H5Dwrite(positions, H5T_NATIVE_DOUBLE, H5S_ALL, H5S_ALL, H5P_DEFAULT, temp);
    H5Dclose(positions);

    hid_t Px = H5Dcreate2(file_id, "Particles_Px", H5T_NATIVE_DOUBLE, dim, H5P_DEFAULT, H5P_DEFAULT, H5P_DEFAULT);
    for(int i = 0; i < particle_aosoa.size(); i++){
        auto part = particle_aosoa.getTuple(i);
        double pos = Cabana::get<part_p>(part,0);
        temp[i] = pos;
    }
    H5Dwrite(Px, H5T_NATIVE_DOUBLE, H5S_ALL, H5S_ALL, H5P_DEFAULT, temp);
    H5Dclose(Px);

    hid_t Py = H5Dcreate2(file_id, "Particles_Py", H5T_NATIVE_DOUBLE, dim, H5P_DEFAULT, H5P_DEFAULT, H5P_DEFAULT);
    for(int i = 0; i < particle_aosoa.size(); i++){
        auto part = particle_aosoa.getTuple(i);
        double pos = Cabana::get<part_p>(part,1);
        temp[i] = pos;
    }
    H5Dwrite(Py, H5T_NATIVE_DOUBLE, H5S_ALL, H5S_ALL, H5P_DEFAULT, temp);
    H5Dclose(Py);

    hid_t Pz = H5Dcreate2(file_id, "Particles_Pz", H5T_NATIVE_DOUBLE, dim, H5P_DEFAULT, H5P_DEFAULT, H5P_DEFAULT);
    for(int i = 0; i < particle_aosoa.size(); i++){
        auto part = particle_aosoa.getTuple(i);
        double pos = Cabana::get<part_p>(part,2);
        temp[i] = pos;
    }
    H5Dwrite(Pz, H5T_NATIVE_DOUBLE, H5S_ALL, H5S_ALL, H5P_DEFAULT, temp);
    H5Dclose(Pz);

    hid_t pmass = H5Dcreate2(file_id, "Particles_mass", H5T_NATIVE_DOUBLE, dim, H5P_DEFAULT, H5P_DEFAULT, H5P_DEFAULT);
    for(int i = 0; i < particle_aosoa.size(); i++){
        auto part = particle_aosoa.getTuple(i);
        temp[i] = Cabana::get<mass>(part);
    }
    H5Dwrite(pmass, H5T_NATIVE_DOUBLE, H5S_ALL, H5S_ALL, H5P_DEFAULT, temp);
    H5Dclose(pmass);

    free(temp);

    temp = (double*)malloc(sizeof(double) * grid_size);
    h_dims[0] = grid_size;
    dim = H5Screate_simple(1, h_dims, NULL);
    //Write the grid
    hid_t Electric_Field_Ex = H5Dcreate2(file_id, "Electric_Field_Ex", H5T_NATIVE_DOUBLE, dim, H5P_DEFAULT, H5P_DEFAULT, H5P_DEFAULT);
    for(int i = 0; i < grid_size; i++){
        temp[i] = ex(i+ng);
    }
    H5Dwrite(Electric_Field_Ex, H5T_NATIVE_DOUBLE, H5S_ALL, H5S_ALL, H5P_DEFAULT, temp);
    H5Dclose(Electric_Field_Ex);

    hid_t Electric_Field_Ey = H5Dcreate2(file_id, "Electric_Field_Ey", H5T_NATIVE_DOUBLE, dim, H5P_DEFAULT, H5P_DEFAULT, H5P_DEFAULT);
    for(int i = 0; i < grid_size; i++){
        temp[i] = ey(i+ng);
    }
    H5Dwrite(Electric_Field_Ey, H5T_NATIVE_DOUBLE, H5S_ALL, H5S_ALL, H5P_DEFAULT, temp);
    H5Dclose(Electric_Field_Ey);

    hid_t Electric_Field_Ez = H5Dcreate2(file_id, "Electric_Field_Ez", H5T_NATIVE_DOUBLE, dim, H5P_DEFAULT, H5P_DEFAULT, H5P_DEFAULT);
    for(int i = 0; i < grid_size; i++){
        temp[i] = ez(i+ng);
    }
    H5Dwrite(Electric_Field_Ez, H5T_NATIVE_DOUBLE, H5S_ALL, H5S_ALL, H5P_DEFAULT, temp);
    H5Dclose(Electric_Field_Ez);

    hid_t Magnetic_Field_Bx = H5Dcreate2(file_id, "Magnetic_Field_Bx", H5T_NATIVE_DOUBLE, dim, H5P_DEFAULT, H5P_DEFAULT, H5P_DEFAULT);
    for(int i = 0; i < grid_size; i++){
        temp[i] = bx(i+ng);
    }
    H5Dwrite(Magnetic_Field_Bx, H5T_NATIVE_DOUBLE, H5S_ALL, H5S_ALL, H5P_DEFAULT, temp);
    H5Dclose(Magnetic_Field_Bx);

    hid_t Magnetic_Field_By = H5Dcreate2(file_id, "Magnetic_Field_By", H5T_NATIVE_DOUBLE, dim, H5P_DEFAULT, H5P_DEFAULT, H5P_DEFAULT);
    for(int i = 0; i < grid_size; i++){
        temp[i] = by(i+ng);
    }
    H5Dwrite(Magnetic_Field_By, H5T_NATIVE_DOUBLE, H5S_ALL, H5S_ALL, H5P_DEFAULT, temp);
    H5Dclose(Magnetic_Field_By);

    hid_t Magnetic_Field_Bz = H5Dcreate2(file_id, "Magnetic_Field_Bz", H5T_NATIVE_DOUBLE, dim, H5P_DEFAULT, H5P_DEFAULT, H5P_DEFAULT);
    for(int i = 0; i < grid_size; i++){
        temp[i] = bz(i+ng);
    }
    H5Dwrite(Magnetic_Field_Bz, H5T_NATIVE_DOUBLE, H5S_ALL, H5S_ALL, H5P_DEFAULT, temp);
    H5Dclose(Magnetic_Field_Bz);

    hid_t Current_Jx = H5Dcreate2(file_id, "Current_Jx", H5T_NATIVE_DOUBLE, dim, H5P_DEFAULT, H5P_DEFAULT, H5P_DEFAULT);
    for(int i = 0; i < grid_size; i++){
        temp[i] = jx(i+ng);
    }
    H5Dwrite(Current_Jx, H5T_NATIVE_DOUBLE, H5S_ALL, H5S_ALL, H5P_DEFAULT, temp);
    H5Dclose(Current_Jx);

    hid_t Current_Jy = H5Dcreate2(file_id, "Current_Jy", H5T_NATIVE_DOUBLE, dim, H5P_DEFAULT, H5P_DEFAULT, H5P_DEFAULT);
    for(int i = 0; i < grid_size; i++){
        temp[i] = jy(i+ng);
    }
    H5Dwrite(Current_Jy, H5T_NATIVE_DOUBLE, H5S_ALL, H5S_ALL, H5P_DEFAULT, temp);
    H5Dclose(Current_Jy);

    hid_t Current_Jz = H5Dcreate2(file_id, "Current_Jz", H5T_NATIVE_DOUBLE, dim, H5P_DEFAULT, H5P_DEFAULT, H5P_DEFAULT);
    for(int i = 0; i < grid_size; i++){
        temp[i] = jz(i+ng);
    }
    H5Dwrite(Current_Jz, H5T_NATIVE_DOUBLE, H5S_ALL, H5S_ALL, H5P_DEFAULT, temp);
    H5Dclose(Current_Jz);

    free(temp);
    H5Fclose(file_id);
}

template<class aosoa, class rank_slice, class id_slice, class weight_slice, class mass_slice,
         class charge_slice, class position_slice, class p_slice>
struct migrate_functor{

    aosoa _particle_aosoa;
    rank_slice rank;
    id_slice id;
    weight_slice weight;
    mass_slice mass;
    charge_slice charge;
    position_slice position;
    p_slice p;
    Kokkos::View<int*, MemorySpace> id_space;
    Kokkos::View<double*, MemorySpace> weight_space;
    Kokkos::View<double*, MemorySpace> mass_space;
    Kokkos::View<double*, MemorySpace> charge_space;
    Kokkos::View<double*, MemorySpace> pos_space;
    Kokkos::View<double*, MemorySpace> px_space;
    Kokkos::View<double*, MemorySpace> py_space;
    Kokkos::View<double*, MemorySpace> pz_space;
    Kokkos::View<int[1], MemorySpace> end;
    Kokkos::View<int[1], MemorySpace> index;
    int myrank;
    int neighbour_rank;


    migrate_functor(aosoa particle_aosoa, Kokkos::View<int*, MemorySpace> _id_space,
                    Kokkos::View<double*, MemorySpace> _weight_space,
                    Kokkos::View<double*, MemorySpace> _mass_space,
                    Kokkos::View<double*, MemorySpace> _charge_space,
                    Kokkos::View<double*, MemorySpace> _pos_space,
                    Kokkos::View<double*, MemorySpace> _px_space,
                    Kokkos::View<double*, MemorySpace> _py_space,
                    Kokkos::View<double*, MemorySpace> _pz_space,
                    rank_slice _rank,
                    id_slice _id,
                    weight_slice _weight,
                    mass_slice _mass,
                    charge_slice _charge,
                    position_slice _position,
                    p_slice _p,
                    int _myrank){
        _particle_aosoa = particle_aosoa;
        id_space = _id_space;
        weight_space = _weight_space;
        mass_space = _mass_space;
        charge_space = _charge_space;
        pos_space = _pos_space;
        px_space = _px_space;
        py_space = _py_space;
        pz_space = _pz_space;
        end = Kokkos::View<int[1], MemorySpace>("end");
        end(0) = particle_aosoa.size();
        index = Kokkos::View<int[1], MemorySpace>("index");
        index(0) = 0;
        rank = _rank;
        id = _id;
        weight = _weight;
        mass = _mass;
        charge = _charge;
        position = _position;
        p = _p;
        myrank = _myrank;
    }

    KOKKOS_INLINE_FUNCTION
    void setup_functor(int _neighbour_rank){
        end(0) = _particle_aosoa.size();
        neighbour_rank = _neighbour_rank;
    }

    KOKKOS_INLINE_FUNCTION
    int get_index(){
        return index(0);
    }

    KOKKOS_INLINE_FUNCTION
    void operator()(const int ix, const int ij) const{
        //TODO
        if( rank.access(ix, ij) == neighbour_rank ){
            int i = Kokkos::atomic_fetch_add(&index(0), int(1));
            id_space(i) = id.access(ix, ij);
            weight_space(i) = weight.access(ix, ij);
            mass_space(i) = mass.access(ix, ij);
            charge_space(i) = charge.access(ix, ij);
            pos_space(i) = position.access(ix, ij, 0);
            px_space(i) = p.access(ix, ij, 0);
            py_space(i) = p.access(ix, ij, 1);
            pz_space(i) = p.access(ix, ij, 2);
            rank.access(ix, ij) = -1;
            //Data saved to the space. Move data from "end" to ix, ij
            // TODO: This doesn't work in parallel for some reason
            // Maybe can do something better with CAS
/*            int myend = Kokkos::atomic_fetch_add(&end(0), int(-1));
            while(Kokkos::atomic_load(&rank(myend)) != myrank){
                myend = Kokkos::atomic_fetch_add(&end(0), int(-1));
            }
            if(&rank(myend) > &rank.access(ix, ij)){
                rank.access(ix, ij) = rank(myend);
                rank(myend) = -1;
                id.access(ix, ij) = id(myend);
                weight.access(ix, ij) = weight(myend);
                mass.access(ix, ij) = mass(myend);
                charge.access(ix, ij) = charge(myend);
                position.access(ix, ij, 0) = position(myend, 0);
                p.access(ix, ij, 0) = p(myend, 0);
                p.access(ix, ij, 1) = p(myend, 1);
                p.access(ix, ij, 2) = p(myend, 2);
            }else{
                rank.access(ix, ij) = -1;
            }*/
        } 
    }
};

template<class aosoa> class Migrator{

    private:
        Kokkos::View<int**, MemorySpace> id_space;
        Kokkos::View<double**, MemorySpace> weight_space;
        Kokkos::View<double**, MemorySpace> mass_space;
        Kokkos::View<double**, MemorySpace> charge_space;
        Kokkos::View<double**, MemorySpace> pos_space;
        Kokkos::View<double**, MemorySpace> px_space;
        Kokkos::View<double**, MemorySpace> py_space;
        Kokkos::View<double**, MemorySpace> pz_space;

    public:
        Migrator(int buffer_size, int nr_neighbours){
            id_space = Kokkos::View<int**, MemorySpace>("temp_id", nr_neighbours, buffer_size);
            weight_space = Kokkos::View<double**, MemorySpace>("temp_weight", nr_neighbours, buffer_size);
            mass_space = Kokkos::View<double**, MemorySpace>("temp_mass", nr_neighbours, buffer_size);
            charge_space = Kokkos::View<double**, MemorySpace>("temp_charge", nr_neighbours, buffer_size);
            pos_space = Kokkos::View<double**, MemorySpace>("temp_pos", nr_neighbours, buffer_size);
            px_space = Kokkos::View<double**, MemorySpace>("temp_px", nr_neighbours, buffer_size);
            py_space = Kokkos::View<double**, MemorySpace>("temp_py", nr_neighbours, buffer_size);
            pz_space = Kokkos::View<double**, MemorySpace>("temp_pz", nr_neighbours, buffer_size);
        }

        void exchange_data( aosoa &particle_aosoa, std::vector<int> neighbors, int myrank, int npart){

            auto rank_slice = Cabana::slice<rank>(particle_aosoa);
 //          Cabana::Distributor<DeviceType> distributor( MPI_COMM_WORLD, rank_slice, neighbors);
           //Initial migration
//           Cabana::migrate( distributor, particle_aosoa );
           
//             Kokkos::View<int* , MemorySpace> id_space("temp_id", npart/10);
//             Kokkos::View<double*, MemorySpace> weight_space("temp_weight", npart/10);
//             Kokkos::View<double*, MemorySpace> mass_space("temp_mass", npart/10);
//             Kokkos::View<double*, MemorySpace> charge_space("temp_charge", npart/10);
//             Kokkos::View<double*, MemorySpace> pos_space("temp_pos", npart/10);
//             Kokkos::View<double*, MemorySpace> px_space("temp_px", npart/10);
//             Kokkos::View<double*, MemorySpace> py_space("temp_py", npart/10);
//             Kokkos::View<double*, MemorySpace> pz_space("temp_pz", npart/10);
             auto id_s = Cabana::slice<id>(particle_aosoa);
             auto weight_s = Cabana::slice<weight>(particle_aosoa);
             auto mass_s = Cabana::slice<mass>(particle_aosoa);
             auto charge_s = Cabana::slice<charge>(particle_aosoa);
             auto part_pos_s = Cabana::slice<part_pos>(particle_aosoa);
             auto part_p_s = Cabana::slice<part_p>(particle_aosoa);
             int *send_count = (int*) malloc(sizeof(int) * neighbors.size());
//             int *send_pos = (int*) malloc(sizeof(int) * neighbors.size());
             int count_neighbours = 0;
             // Need to find a way to make this parallel
             // Ideally atomic free & parallel...
             // First easy thing to do is have one buffer per
             // neighbour instead of one total buffer, so when
             // finding a one to send we can just move it instead of
             // ignoring it unless it is neighbour X. For our current
             // testcase (2 ranks) this actually does nothing though.
             //
             // Atomic method:
             // When find one:
             // 1. myindex = atomicAdd(&count_neighbours, 1);  // Check if this gives before or after result.
             // 2. Update id_space etc. for myindex
             // 3. myend = atomicAdd(&end, -1);
             // while rank_slice(myend) != myrank && rank_slice(myend) >= 0
             // myend = atomicAdd(&end, -1);
             // Copy myend info into current index. Set rank of myend to -1
             // Essentially lock free DEQ
             //
             // Expectation is that low numbers of particles will move nodes.
             // Low atomic collision rate ==> Probably reasonable performance/scaling.
             // Initial implementation check: Contiguous block of rank == myrank then
             // -1s
             // One buffer, index, end per neighbour may lead to better performance (one loop)
             // for more memory tradeoff. Even fewer collisions on myindex though
             /*migrate_functor<decltype(particle_aosoa),
                             decltype(rank_slice),
                             decltype(id_s),
                             decltype(weight_s),
                             decltype(mass_s),
                             decltype(charge_s),
                             decltype(part_pos_s),
                             decltype(part_p_s)> mig_func(
                                particle_aosoa, id_space, weight_space, mass_space,
                                charge_space, pos_space, px_space, py_space,
                                pz_space, rank_slice, id_s, weight_s,
                                mass_s, charge_s, part_pos_s, part_p_s, myrank);*/
             //Cabana::SimdPolicy<VectorLength, ExecutionSpace> simd_policy( 0,
               //                               particle_aosoa.size());
             int end = particle_aosoa.size() -1;
             for(int i = 0; i < neighbors.size(); i++){
                send_count[i] = 0;
             }
             for(int i = 0; i < particle_aosoa.size(); i++){
                if(rank_slice(i) != myrank && rank_slice(i) >= 0){
                    int therank = rank_slice(i);
                    for(int k = 0; k < neighbors.size(); k++){
                        if(therank == neighbors[k]){
                            therank = k;
                            break;
                        }
                    }
                    int pos = send_count[therank];
                    id_space(therank, pos) = id_s(i);
                    weight_space(therank, pos) = weight_s(i);
                    mass_space(therank, pos) = mass_s(i);
                    charge_space(therank, pos) = charge_s(i);
                    pos_space(therank, pos) = part_pos_s(i, 0);
                    px_space(therank, pos) = part_p_s(i, 0);
                    py_space(therank, pos) = part_p_s(i, 1);
                    pz_space(therank, pos) = part_p_s(i, 2);
                    send_count[therank]++;

                    //Move from end
                    while(rank_slice(end) != myrank){
                        end--;
                    }
                    if(end > i){
                        //Copy from end
                        rank_slice(i) = rank_slice(end);
                        id_s(i) = id_s(end);
                        weight_s(i) = weight_s(end);
                        mass_s(i) = mass_s(end);
                        charge_s(i) = charge_s(end);
                        part_pos_s(i,0) = part_pos_s(end, 0);
                        part_p_s(i, 0) = part_p_s(end, 0);
                        part_p_s(i, 1) = part_p_s(end, 1);
                        part_p_s(i, 2) = part_p_s(end, 2);
                        rank_slice(end) = -1;
                    }else{
                        rank_slice(i) = -1;
                        end++;
                    }
                }   
             } 
//             for(int i = 0; i < neighbors.size(); i++){
//                send_pos[i] = count_neighbours;
//                int neighbour = neighbors[i];
//                mig_func.setup_functor(neighbour);
//                if(neighbour==myrank){send_count[i] = 0; continue;}
//                Cabana::simd_parallel_for(simd_policy, mig_func ,"migration_functor");
//                Kokkos::fence();
                /*for(int i = 0; i < particle_aosoa.size(); i++){
                    if(rank_slice(i) == neighbour)
                    {
                        id_space(count_neighbours) = id_s(i); 
                        weight_space(count_neighbours) = weight_s(i); 
                        mass_space(count_neighbours) = mass_s(i); 
                        charge_space(count_neighbours) = charge_s(i); 
                        pos_space(count_neighbours) = part_pos_s(i,0);
                        px_space(count_neighbours) = part_p_s(i,0);
                        py_space(count_neighbours) = part_p_s(i,1);
                        pz_space(count_neighbours) = part_p_s(i,2);
                        rank_slice(i) = -1;
                        count_neighbours++;
                    }
                }*/
//                count_neighbours = mig_func.get_index();
//                send_count[i] = count_neighbours - send_pos[i];
//           }
//           std::cout << send_count[0] << ", " << send_count[1] << ".\n";
/*           for(int i = 1; i < particle_aosoa.size(); i++){
                if(rank_slice(i) == myrank){
                    if(rank_slice(i-1) != myrank){
                        std::cout << "Position " << i << "was at " << myrank << "but previous was at " << rank_slice(i-1) << "\n";
                    }
                    assert(rank_slice(i-1) == myrank);
                    assert(rank_slice(i-1) != (1-myrank));
                }
                if(particle_aosoa.size() - i < 53){
                    assert(rank_slice(i) == -1);
                }
           }*/
           // Data collected, need to send information to neighbours to know what to expect
           int *recv_count = (int*) malloc(sizeof(int) * neighbors.size());
//           int *recv_pos = (int*) malloc(sizeof(int)  * neighbors.size());
           MPI_Request *requests = (MPI_Request*) malloc(sizeof(MPI_Request) * neighbors.size() * 2);
           int req_num = 0;
           for(int i = 0; i < neighbors.size(); i++){
                    recv_count[i] = 0;
                if(neighbors[i] == myrank){
                    continue;
                }
                MPI_Irecv(&recv_count[i], 1, MPI_INT, neighbors[i], 0, MPI_COMM_WORLD, &requests[req_num++]);
                MPI_Isend(&send_count[i], 1, MPI_INT, neighbors[i], 0, MPI_COMM_WORLD, &requests[req_num++]);
           }
           MPI_Waitall(req_num, requests, MPI_STATUSES_IGNORE);
           MPI_Barrier(MPI_COMM_WORLD);
//           recv_pos[0] = 0;
           int total_size = 0;
           for(int i = 0; i < neighbors.size(); i++){
//                recv_pos[i] = recv_pos[i-1] + recv_count[i-1];
                total_size += recv_count[i];
           }
  //         int total_size = recv_pos[neighbors.size()-1] + recv_count[neighbors.size()-1];

           //Construct buffers
           Kokkos::View<int** , MemorySpace> r_id_space("temp_id", neighbors.size(), total_size);
           Kokkos::View<double**, MemorySpace> r_weight_space("temp_weight", neighbors.size(), total_size);
           Kokkos::View<double**, MemorySpace> r_mass_space("temp_mass", neighbors.size(), total_size);
           Kokkos::View<double**, MemorySpace> r_charge_space("temp_charge", neighbors.size(), total_size);
           Kokkos::View<double**, MemorySpace> r_pos_space("temp_pos", neighbors.size(), total_size);
           Kokkos::View<double**, MemorySpace> r_px_space("temp_px", neighbors.size(), total_size);
           Kokkos::View<double**, MemorySpace> r_py_space("temp_py", neighbors.size(), total_size);
           Kokkos::View<double**, MemorySpace> r_pz_space("temp_pz", neighbors.size(), total_size);

           free(requests);
           requests = (MPI_Request*) malloc(sizeof(MPI_Request) * neighbors.size() * 2 * 8);
           req_num = 0;
           for(int i = 0; i < neighbors.size(); i++){
                //Time to send & recv data
                if(neighbors[i] != myrank){
                    MPI_Irecv(&r_id_space(i,0), recv_count[i], MPI_INT, neighbors[i], 0, MPI_COMM_WORLD, &requests[req_num++]);
                    MPI_Irecv(&r_weight_space(i,0), recv_count[i], MPI_DOUBLE, neighbors[i], 1, MPI_COMM_WORLD, &requests[req_num++]);
                    MPI_Irecv(&r_mass_space(i,0), recv_count[i], MPI_DOUBLE, neighbors[i], 2, MPI_COMM_WORLD, &requests[req_num++]);
                    MPI_Irecv(&r_charge_space(i,0), recv_count[i], MPI_DOUBLE, neighbors[i], 3, MPI_COMM_WORLD, &requests[req_num++]);
                    MPI_Irecv(&r_pos_space(i,0), recv_count[i], MPI_DOUBLE, neighbors[i], 4, MPI_COMM_WORLD, &requests[req_num++]);
                    MPI_Irecv(&r_px_space(i,0), recv_count[i], MPI_DOUBLE, neighbors[i], 5, MPI_COMM_WORLD, &requests[req_num++]);
                    MPI_Irecv(&r_py_space(i,0), recv_count[i], MPI_DOUBLE, neighbors[i], 6, MPI_COMM_WORLD, &requests[req_num++]);
                    MPI_Irecv(&r_pz_space(i,0), recv_count[i], MPI_DOUBLE, neighbors[i], 7, MPI_COMM_WORLD, &requests[req_num++]);

                    MPI_Isend(&id_space(i,0), send_count[i], MPI_INT, neighbors[i], 0, MPI_COMM_WORLD, &requests[req_num++]);
                    MPI_Isend(&weight_space(i,0), send_count[i], MPI_DOUBLE, neighbors[i], 1, MPI_COMM_WORLD, &requests[req_num++]);
                    MPI_Isend(&mass_space(i,0), send_count[i], MPI_DOUBLE, neighbors[i], 2, MPI_COMM_WORLD, &requests[req_num++]);
                    MPI_Isend(&charge_space(i,0), send_count[i], MPI_DOUBLE, neighbors[i], 3, MPI_COMM_WORLD, &requests[req_num++]);
                    MPI_Isend(&pos_space(i,0), send_count[i], MPI_DOUBLE, neighbors[i], 4, MPI_COMM_WORLD, &requests[req_num++]);
                    MPI_Isend(&px_space(i,0), send_count[i], MPI_DOUBLE, neighbors[i], 5, MPI_COMM_WORLD, &requests[req_num++]);
                    MPI_Isend(&py_space(i,0), send_count[i], MPI_DOUBLE, neighbors[i], 6, MPI_COMM_WORLD, &requests[req_num++]);
                    MPI_Isend(&pz_space(i,0), send_count[i], MPI_DOUBLE, neighbors[i], 7, MPI_COMM_WORLD, &requests[req_num++]);
                }
           }
           MPI_Waitall(req_num, requests, MPI_STATUSES_IGNORE);
           free(requests);
           //Data has arrived, need to put it into AOSOA
           
           int recvd = 0;//recv_pos[neighbors.size()-1] + recv_count[neighbors.size()-1];
           int sent = 0; //send_pos[neighbors.size()-1] + send_count[neighbors.size()-1];
           for(int i = 0; i < neighbors.size(); i++){
                recvd += recv_count[i];
                sent += send_count[i];
           }
           int size_change = recvd - sent;
           //If size_change < 0 we lost particles, else we gained.
           //Move all the sent particles to the end
           /*while(rank_slice(end) == -1 && end > 0){
                end--;
           }
           for(int i = 0; i < end; i++){
                if(rank_slice(i) == -1){
                    int id = id_s(end);
                    double weight = weight_s(end);
                    double mass = mass_s(end);
                    double charge = charge_s(end);
                    double part_pos = part_pos_s(end,0);
                    double px = part_p_s(end, 0);
                    double py = part_p_s(end, 1);
                    double pz = part_p_s(end, 2);
                    int rank = rank_slice(end);

                    rank_slice(end) = -1;

                    rank_slice(i) = rank;
                    id_s(i) = id;
                    weight_s(i) = weight;
                    mass_s(i) = mass;
                    charge_s(i) = charge;
                    part_pos_s(i, 0) = part_pos;
                    part_p_s(i, 0) = px;
                    part_p_s(i, 1) = py;
                    part_p_s(i, 2) = pz;

                    while(rank_slice(end) == -1 && end > 0){
                        end--;
                    }

                }            
           }*/
           int current_size = particle_aosoa.size();
           if(size_change != 0){
            particle_aosoa.resize(current_size+size_change);
           }
           auto new_rank_slice = Cabana::slice<rank>(particle_aosoa);
            if(size_change > 0){
                if(sent == 0){
                    end = current_size;
                }
                for(int i = end; i < particle_aosoa.size(); i++){
                    new_rank_slice(i) = -1;
                }
            }
           auto new_id_s = Cabana::slice<id>(particle_aosoa);
           auto new_weight_s = Cabana::slice<weight>(particle_aosoa);
           auto new_mass_s = Cabana::slice<mass>(particle_aosoa);
           auto new_charge_s = Cabana::slice<charge>(particle_aosoa);
           auto new_part_pos_s = Cabana::slice<part_pos>(particle_aosoa);
           auto new_part_p_s = Cabana::slice<part_p>(particle_aosoa);
           int x = 0;
           for(int j = 0; j < neighbors.size(); j++){
                for(int i = 0; i < recv_count[j]; i++){
//                     if(myrank == 1 && r_id_space(j,i) == 1424990){   
//                        std::cout << "Receieved ID 1424990 at " << j << ", " << i << " with position " << r_pos_space(j,i) << "\n" << std::flush;
//                     }
                     assert(new_rank_slice(end+x) == -1);
                     new_id_s(end+x) = r_id_space(j,i);
                     new_weight_s(end+x) = r_weight_space(j,i);
                     new_mass_s(end+x) = r_mass_space(j,i);
                     new_charge_s(end+x) = r_charge_space(j,i);
                     //assert(r_pos_space(j,i) == r_pos_space(j,i));
                     new_part_pos_s(end+x,0) = r_pos_space(j,i);
                     new_part_p_s(end+x,0) = r_px_space(j,i);
                     new_part_p_s(end+x,1) = r_py_space(j,i);
                     new_part_p_s(end+x,2) = r_pz_space(j,i);
                     new_rank_slice(end+x) = myrank;
                     x++;
                }
           }
        free(recv_count);
//        free(recv_pos);
        free(send_count);
//        free(send_pos);
//        std::cout << "Last updated was " << end+x-1 << ". AoSoA size is " << particle_aosoa.size() << "\n";
           
        //TODO Check somehow all positions/values are "nice"
        for(int i = 0; i < particle_aosoa.size(); i++){
//            if(new_part_pos_s(i, 0) < 0.0){
//                std::cout << "found particle at " << new_part_pos_s(i, 0) << " with id " << new_id_s(i) << "\n";
//            }
            assert(new_part_pos_s(i,0) >= 0.0);
            assert(new_part_pos_s(i,0) <= 2.0);
            assert(new_part_pos_s(i,0) == new_part_pos_s(i,0));
            assert(new_rank_slice(i) == myrank);
        }
           
           
    }
};

int main(int argc, char* argv[] ){
    //TODO
    
    Kokkos::ScopeGuard scope_guard(argc, argv);
    int provided;
    MPI_Init_thread( &argc, &argv, MPI_THREAD_FUNNELED, &provided  );
//    Kokkos::initialize(argc,argv);
    {
        int step = 0;
        double time = 0.0;
    
        struct config_type config;
        // TODO MPI INIT ignore
        config.field = field_struct_type("field", 1);
        //struct field field;
        //TODO Minimal init
        auto host_field = Kokkos::create_mirror_view(config.field);
        
        int nxglobal = 28500;
        int npart_global = nxglobal * 100;
        int myrank; MPI_Comm_rank( MPI_COMM_WORLD, &myrank );
        int nranks; MPI_Comm_size( MPI_COMM_WORLD, &nranks );
        int npart = 0;
        // This can be done better but it'll do for now.
        if(npart_global % nranks == 0){
            npart = npart_global / nranks;
        }else{
            npart = npart_global / nranks;
            if(npart_global % nranks > myrank){
                npart++;
            }
        }
        //TODO: Compute local cell IDs
        int nxlocal = 0;
        if(nxglobal % nranks == 0){
            nxlocal = nxglobal / nranks;
        }else{
            nxlocal = nxglobal / nranks;
            if(nxglobal % nranks > myrank){
                nxlocal++;
            }
        }
        int min_local_cell = (myrank * (nxglobal / nranks));
        if(nxglobal % nranks != 0){
            if( myrank > nxglobal % nranks ){
                min_local_cell += (nxglobal % nranks);
            }else{
                min_local_cell += (myrank-1);
            }
        }
        //Exclusive
        int max_local_cell = min_local_cell + nxlocal;
        double x_min = 0.0;
        double x_max = 2.0;
        double dx = 2.0 / (double)(nxglobal);
        int ng = 4; // number ghost cells
        int jng = 4; // number current ghost cells
        //Setup boundary struct
        boundary box;
        box.x_min = 0.0;
        box.x_max = 2.0;
        double cell_size = (box.x_max - box.x_min) / (double)(nxglobal);

        //Compute local grid
        double x_grid_min_local = 3.5087719298245611E-005 + min_local_cell * cell_size;
        double x_grid_max_local = max_local_cell * cell_size;
        minimal_init(host_field, x_grid_min_local, x_grid_max_local);
        
        config.ex = field_type("ex", nxlocal + 2*ng);
        config.ey = field_type("ey", nxlocal + 2*ng);
        config.ez = field_type("ez", nxlocal + 2*ng);
        config.bx = field_type("bx", nxlocal + 2*ng);
        config.by = field_type("by", nxlocal + 2*ng);
        config.bz = field_type("bz", nxlocal + 2*ng);
        config.jx = field_type("jx", nxlocal + 2*jng);
        config.jy = field_type("jy", nxlocal + 2*jng);
        config.jz = field_type("jz", nxlocal + 2*jng);
       
        std::cout << "[" << myrank << "]" << config.ex.size() << "\n"; 
    
        //config.ex = field_type("ex", nxglobal + 2*ng);
        //config.ey = field_type("ey", nxglobal + 2*ng);
        //config.ez = field_type("ez", nxglobal + 2*ng);
        //config.bx = field_type("bx", nxglobal + 2*ng);
        //config.by = field_type("by", nxglobal + 2*ng);
        //config.bz = field_type("bz", nxglobal + 2*ng);
        //config.jx = field_type("jx", nxglobal + 2*jng);
        //config.jy = field_type("jy", nxglobal + 2*jng);
        //config.jz = field_type("jz", nxglobal + 2*jng);

        //Actual positions go from ng to ng + nxglobal -1 inclusive
    
        // Make host copies for initialisation
        auto host_ex = Kokkos::create_mirror_view(config.ex);
        auto host_ey = Kokkos::create_mirror_view(config.ey);
        auto host_ez = Kokkos::create_mirror_view(config.ez);
        auto host_bx = Kokkos::create_mirror_view(config.bx);
        auto host_by = Kokkos::create_mirror_view(config.by);
        auto host_bz = Kokkos::create_mirror_view(config.bz);
        auto host_jx = Kokkos::create_mirror_view(config.jx);
        auto host_jy = Kokkos::create_mirror_view(config.jy);
        auto host_jz = Kokkos::create_mirror_view(config.jz);

    
/*        Kokkos::parallel_for("Init e&b arrays", nxglobal+2*ng, 
                KOKKOS_LAMBDA( const int &ix ){
                    host_ex[ix] = 0.0;
                    host_ey[ix] = 0.0;
                    host_ez[ix] = 0.0;
                    host_bx[ix] = 0.0;
                    host_by[ix] = 0.0;
                    host_bz[ix] = 0.0;
        });
        Kokkos::parallel_for("Init j arrays", nxglobal + 2*jng,
                KOKKOS_LAMBDA(const int &ix){
                host_jx[ix] = 0.0;
                host_jy[ix] = 0.0;
                host_jz[ix] = 0.0;
        });*/
    
    
        //Set up particles
    
        Cabana::AoSoA<DataTypes, DeviceType, VectorLength> particle_aosoa( "particle_list",
                                                                           npart);
        Cabana::AoSoA<DataTypes, HostType, VectorLength> host_particle_aosoa( "particles_host",
                                                                              npart);
    
   
        
        after_control(host_ex, host_ey, host_ez, host_bx, host_by,
                      host_bz, host_jx, host_jy, host_jz, nxlocal,
                      ng, jng);
        
        //after_control(host_ex, host_ey, host_ez, host_bx, host_by,
        //              host_bz, host_jx, host_jy, host_jz, nxglobal,
        //              ng, jng);
        //open_files setup.f90 -- NYI for opening files
        //
        //
        //Rescan input deck for things that required allocating
        //read_deck
        //after_deck_last
        //
        //if restart - ignore
        //
        //pre_load_balance //These two are TODO - Nothing to do on one node.

        std::cout << "cell_size = " << cell_size << "\n";
        auto_load(host_particle_aosoa, box, npart, nxglobal, cell_size, nxglobal, dx, myrank, nranks,
               min_local_cell, max_local_cell );
        time = 0.0;

        double dt = host_field(0).cfl * dx / c;
        std::cout << "dt was " << dt << "\n";
   
        //TODO efield_bcs
        //Deep copy to main fields
        Kokkos::deep_copy(config.ex, host_ex);
        Kokkos::deep_copy(config.ey, host_ey);
        Kokkos::deep_copy(config.ez, host_ez);
        Kokkos::deep_copy(config.bx, host_bx);
        Kokkos::deep_copy(config.by, host_by);
        Kokkos::deep_copy(config.bz, host_bz);
        Kokkos::deep_copy(config.jx, host_jx);
        Kokkos::deep_copy(config.jy, host_jy);
        Kokkos::deep_copy(config.jz, host_jz);
        Kokkos::deep_copy(config.field, host_field);
        Cabana::deep_copy(particle_aosoa, host_particle_aosoa);
        efield_bcs(config.ex, config.ey, config.ez, nxglobal, ng);
        double dt_store = dt;
        dt = dt / 2.0;
        time = time + dt;
        bfield_final_bcs(config.bx, config.by, config.bz, nxglobal, ng);
        dt = dt_store;

        double particle_push_start_time = 0.0;
        double t_end = 6.2132e-9;
//        double t_end = 6.2132e-9 / 20.0;
//        double t_end = dt * 120.0;
        std::cout << "dt = " << dt << "\n";
        int nsteps = -1;
        bool halt = false;
        //double per_dump = t_end / 20.0;
        double per_dump = t_end / 20.0;
        double next_dump = per_dump;
        int dump_count = 0;
        //Deep copy to host fields
        Kokkos::deep_copy(host_ex, config.ex);
        Kokkos::deep_copy(host_ey, config.ey);
        Kokkos::deep_copy(host_ez, config.ez);
        Kokkos::deep_copy(host_bx, config.bx);
        Kokkos::deep_copy(host_by, config.by);
        Kokkos::deep_copy(host_bz, config.bz);
        Kokkos::deep_copy(host_jx, config.jx);
        Kokkos::deep_copy(host_jy, config.jy);
        output_routines(host_particle_aosoa, host_ex, host_ey, host_ez,
                        host_bx, host_by, host_bz, host_jx, host_jy, host_jz,
                        dump_count, nxlocal, ng, jng);
        //TODO output_routines(pic_system, ex, ey, ez, bx, by, bz, jx, jy, jz, dump_count, nxglobal);
        dump_count++;

        

        //Create the functors and other stuff
        update_e_field_functor update_e_field(config/*, nxglobal*/);

        update_b_field_functor update_b_field(config/*, nxglobal*/);
       
        
        Kokkos::RangePolicy<> rangepolicy(ng, nxlocal+ng+1);
         
//        Kokkos::RangePolicy<> rangepolicy(ng, nxglobal+ng+1);

//        auto id_s = Cabana::slice<id>(particle_aosoa);
//        auto weight_s = Cabana::slice<weight>(particle_aosoa);
//        auto mass_s = Cabana::slice<mass>(particle_aosoa);
//        auto charge_s = Cabana::slice<charge>(particle_aosoa);
//        auto part_pos_s = Cabana::slice<part_pos>(particle_aosoa);
//        auto part_p_s = Cabana::slice<part_p>(particle_aosoa);
//        auto rank_slice = Cabana::slice<rank>(particle_aosoa);

       Cabana::SimdPolicy<VectorLength, ExecutionSpace> simd_policy( 0,
                                              particle_aosoa.size());
       int previous_rank = ( myrank == 0 ) ? nranks - 1 : myrank - 1;
       int next_rank = ( myrank == nranks - 1 ) ? 0 : myrank + 1;
       std::vector<int> neighbors = { previous_rank, myrank, next_rank };
       std::sort( neighbors.begin(), neighbors.end() );
       auto unique_end = std::unique( neighbors.begin(), neighbors.end() );
       neighbors.resize( std::distance( neighbors.begin(), unique_end ) );

       Migrator<decltype(particle_aosoa)> migrator(particle_aosoa.size() / 10, neighbors.size());
       {
           if(nranks > 1){
           migrator.exchange_data( particle_aosoa, neighbors, myrank, particle_aosoa.size());
           }
//            auto rank_slice = Cabana::slice<rank>(particle_aosoa);
//           Cabana::Distributor<DeviceType> distributor( MPI_COMM_WORLD, rank_slice, neighbors);
           //Initial migration
//           Cabana::migrate( distributor, particle_aosoa );
           
/*             Kokkos::View<int* , MemorySpace> id_space("temp_id", npart/10);
             Kokkos::View<double*, MemorySpace> weight_space("temp_weight", npart/10);
             Kokkos::View<double*, MemorySpace> mass_space("temp_mass", npart/10);
             Kokkos::View<double*, MemorySpace> charge_space("temp_charge", npart/10);
             Kokkos::View<double*, MemorySpace> pos_space("temp_pos", npart/10);
             Kokkos::View<double*, MemorySpace> px_space("temp_px", npart/10);
             Kokkos::View<double*, MemorySpace> py_space("temp_py", npart/10);
             Kokkos::View<double*, MemorySpace> pz_space("temp_pz", npart/10);
             auto id_s = Cabana::slice<id>(particle_aosoa);
             auto weight_s = Cabana::slice<weight>(particle_aosoa);
             auto mass_s = Cabana::slice<mass>(particle_aosoa);
             auto charge_s = Cabana::slice<charge>(particle_aosoa);
             auto part_pos_s = Cabana::slice<part_pos>(particle_aosoa);
             auto part_p_s = Cabana::slice<part_p>(particle_aosoa);
             int *send_count = (int*) malloc(sizeof(int) * neighbors.size());
             int *send_pos = (int*) malloc(sizeof(int) * neighbors.size());
             int count_neighbours = 0;
             for(int i = 0; i < neighbors.size(); i++){
                send_pos[i] = count_neighbours;
                int neighbour = neighbors[i];
                if(neighbour==myrank){send_count[i] = 0; continue;}
                for(int i = 0; i < particle_aosoa.size(); i++){
                    if(rank_slice(i) == neighbour)
                    {
                        id_space(count_neighbours) = id_s(i); 
                        weight_space(count_neighbours) = weight_s(i); 
                        mass_space(count_neighbours) = mass_s(i); 
                        charge_space(count_neighbours) = charge_s(i); 
                        pos_space(count_neighbours) = part_pos_s(i,0);
                        px_space(count_neighbours) = part_p_s(i,0);
                        py_space(count_neighbours) = part_p_s(i,1);
                        pz_space(count_neighbours) = part_p_s(i,2);
                        rank_slice(i) = -1;
                        count_neighbours++;
                    }
                }
                send_count[i] = count_neighbours - send_pos[i];
           }
           // Data collected, need to send information to neighbours to know what to expect
           int *recv_count = (int*) malloc(sizeof(int) * neighbors.size());
           int *recv_pos = (int*) malloc(sizeof(int)  * neighbors.size());
           MPI_Request *requests = (MPI_Request*) malloc(sizeof(MPI_Request) * neighbors.size() * 2);
           int req_num = 0;
           for(int i = 0; i < neighbors.size(); i++){
                    recv_count[i] = 0;
                if(neighbors[i] == myrank){
                    continue;
                }
                MPI_Irecv(&recv_count[i], 1, MPI_INT, neighbors[i], 0, MPI_COMM_WORLD, &requests[req_num++]);
                MPI_Isend(&send_count[i], 1, MPI_INT, neighbors[i], 0, MPI_COMM_WORLD, &requests[req_num++]);
           }
           MPI_Waitall(req_num, requests, MPI_STATUSES_IGNORE);
           MPI_Barrier(MPI_COMM_WORLD);
           recv_pos[0] = 0;
           for(int i = 1; i < neighbors.size(); i++){
                recv_pos[i] = recv_pos[i-1] + recv_count[i-1];
           }
           int total_size = recv_pos[neighbors.size()-1] + recv_count[neighbors.size()-1];

           //Construct buffers
           Kokkos::View<int* , MemorySpace> r_id_space("temp_id", total_size);
           Kokkos::View<double*, MemorySpace> r_weight_space("temp_weight", total_size);
           Kokkos::View<double*, MemorySpace> r_mass_space("temp_mass", total_size);
           Kokkos::View<double*, MemorySpace> r_charge_space("temp_charge", total_size);
           Kokkos::View<double*, MemorySpace> r_pos_space("temp_pos", total_size);
           Kokkos::View<double*, MemorySpace> r_px_space("temp_px", total_size);
           Kokkos::View<double*, MemorySpace> r_py_space("temp_py", total_size);
           Kokkos::View<double*, MemorySpace> r_pz_space("temp_pz", total_size);

           free(requests);
           requests = (MPI_Request*) malloc(sizeof(MPI_Request) * neighbors.size() * 2 * 8);
           req_num = 0;
           for(int i = 0; i < neighbors.size(); i++){
                //Time to send & recv data
                MPI_Irecv(&r_id_space.data()[recv_pos[i]], recv_count[i], MPI_INT, neighbors[i], 0, MPI_COMM_WORLD, &requests[req_num++]);
                MPI_Irecv(&r_weight_space.data()[recv_pos[i]], recv_count[i], MPI_DOUBLE, neighbors[i], 1, MPI_COMM_WORLD, &requests[req_num++]);
                MPI_Irecv(&r_mass_space.data()[recv_pos[i]], recv_count[i], MPI_DOUBLE, neighbors[i], 2, MPI_COMM_WORLD, &requests[req_num++]);
                MPI_Irecv(&r_charge_space.data()[recv_pos[i]], recv_count[i], MPI_DOUBLE, neighbors[i], 3, MPI_COMM_WORLD, &requests[req_num++]);
                MPI_Irecv(&r_pos_space.data()[recv_pos[i]], recv_count[i], MPI_DOUBLE, neighbors[i], 4, MPI_COMM_WORLD, &requests[req_num++]);
                MPI_Irecv(&r_px_space.data()[recv_pos[i]], recv_count[i], MPI_DOUBLE, neighbors[i], 5, MPI_COMM_WORLD, &requests[req_num++]);
                MPI_Irecv(&r_py_space.data()[recv_pos[i]], recv_count[i], MPI_DOUBLE, neighbors[i], 6, MPI_COMM_WORLD, &requests[req_num++]);
                MPI_Irecv(&r_pz_space.data()[recv_pos[i]], recv_count[i], MPI_DOUBLE, neighbors[i], 7, MPI_COMM_WORLD, &requests[req_num++]);

                MPI_Isend(&id_space.data()[send_pos[i]], send_count[i], MPI_INT, neighbors[i], 0, MPI_COMM_WORLD, &requests[req_num++]);
                MPI_Isend(&weight_space.data()[send_pos[i]], send_count[i], MPI_DOUBLE, neighbors[i], 1, MPI_COMM_WORLD, &requests[req_num++]);
                MPI_Isend(&mass_space.data()[send_pos[i]], send_count[i], MPI_DOUBLE, neighbors[i], 2, MPI_COMM_WORLD, &requests[req_num++]);
                MPI_Isend(&charge_space.data()[send_pos[i]], send_count[i], MPI_DOUBLE, neighbors[i], 3, MPI_COMM_WORLD, &requests[req_num++]);
                MPI_Isend(&pos_space.data()[send_pos[i]], send_count[i], MPI_DOUBLE, neighbors[i], 4, MPI_COMM_WORLD, &requests[req_num++]);
                MPI_Isend(&px_space.data()[send_pos[i]], send_count[i], MPI_DOUBLE, neighbors[i], 5, MPI_COMM_WORLD, &requests[req_num++]);
                MPI_Isend(&py_space.data()[send_pos[i]], send_count[i], MPI_DOUBLE, neighbors[i], 6, MPI_COMM_WORLD, &requests[req_num++]);
                MPI_Isend(&pz_space.data()[send_pos[i]], send_count[i], MPI_DOUBLE, neighbors[i], 7, MPI_COMM_WORLD, &requests[req_num++]);
           }
           MPI_Waitall(req_num, requests, MPI_STATUSES_IGNORE);
           free(requests);
           //Data has arrived, need to put it into AOSOA
           int recvd = recv_pos[neighbors.size()-1] + recv_count[neighbors.size()-1];
           int sent = send_pos[neighbors.size()-1] + send_count[neighbors.size()-1];
           int size_change = recvd - sent;
           //If size_change < 0 we lost particles, else we gained.
           //Move all the sent particles to the end
           int end = particle_aosoa.size() -1;
           while(rank_slice(end) == -1 && end > 0){
                end--;
           }
           for(int i = 0; i < end; i++){
                if(rank_slice(i) == -1){
                    int id = id_s(end);
                    double weight = weight_s(end);
                    double mass = mass_s(end);
                    double charge = charge_s(end);
                    double part_pos = part_pos_s(end,0);
                    double px = part_p_s(end, 0);
                    double py = part_p_s(end, 1);
                    double pz = part_p_s(end, 2);
                    int rank = rank_slice(end);

                    rank_slice(end) = -1;

                    rank_slice(i) = rank;
                    id_s(i) = id;
                    weight_s(i) = weight;
                    mass_s(i) = mass;
                    charge_s(i) = charge;
                    part_pos_s(i, 0) = part_pos;
                    part_p_s(i, 0) = px;
                    part_p_s(i, 1) = py;
                    part_p_s(i, 2) = pz;

                    while(rank_slice(end) == -1 && end > 0){
                        end--;
                    }

                }            
           }
           int current_size = particle_aosoa.size();
           particle_aosoa.resize(current_size+size_change);
           id_s = Cabana::slice<id>(particle_aosoa);
           weight_s = Cabana::slice<weight>(particle_aosoa);
           mass_s = Cabana::slice<mass>(particle_aosoa);
           charge_s = Cabana::slice<charge>(particle_aosoa);
           part_pos_s = Cabana::slice<part_pos>(particle_aosoa);
           part_p_s = Cabana::slice<part_p>(particle_aosoa);
           for(int i = 0; i < recvd; i++){
                id_s(end+i) = r_id_space(i);
                weight_s(end+i) = r_weight_space(i);
                mass_s(end+i) = r_mass_space(i);
                charge_s(end+i) = r_charge_space(i);
                part_pos_s(end+i,0) = r_pos_space(i);
                part_p_s(end+i,0) = r_px_space(i);
                part_p_s(end+i,1) = r_py_space(i);
                part_p_s(end+i,2) = r_pz_space(i);
           }
        free(recv_count);
        free(recv_pos);
        free(send_count);
        free(send_pos);*/
           
           
           
           
           
       }

       Kokkos::Timer timer;
       double field_solve_time = 0.0;
       double particle_push_time = 0.0;
       double field_solve_start;
       double field_solve_end;
       double particle_start;
       double particle_end;
        //TODO timestepping loop
        while(true){
           if ((step >= nsteps and nsteps >= 0) || time >= t_end || halt ){break;}
            auto id_s = Cabana::slice<id>(particle_aosoa);
            auto weight_s = Cabana::slice<weight>(particle_aosoa);
            auto mass_s = Cabana::slice<mass>(particle_aosoa);
            auto charge_s = Cabana::slice<charge>(particle_aosoa);
            auto part_pos_s = Cabana::slice<part_pos>(particle_aosoa);
            auto part_p_s = Cabana::slice<part_p>(particle_aosoa);
            auto rank_slice = Cabana::slice<rank>(particle_aosoa);
            Cabana::SimdPolicy<VectorLength, ExecutionSpace> simd_policy( 0,
                                                      particle_aosoa.size());
            particle_bcs_functor<decltype(part_pos_s), decltype(rank_slice)> pbf(box, part_pos_s, rank_slice);

            push_particles_functor<decltype(id_s), decltype(weight_s), decltype(mass_s),
                                   decltype(charge_s), decltype(part_pos_s), decltype(part_p_s)> 
                                       push_particles(id_s, weight_s, mass_s, charge_s, part_pos_s, part_p_s,
                                                      dt, dx, config, nxglobal, ng, jng );
           bool push = (time >= particle_push_start_time);
           field_solve_start = timer.seconds();
           update_eb_fields_half(config.ex, config.ey, config.ez, nxglobal, config.jx, config.jy, config.jz, 
                   config.bx, config.by, config.bz,
                   dt, dx, host_field, config.field, ng, update_e_field, update_b_field, rangepolicy);
           field_solve_end = timer.seconds();
           field_solve_time += (field_solve_end - field_solve_start);
           if(push){
                //Reset the current
                current_start(config.jx, config.jy, config.jz, nxlocal, jng);
                //current_start(config.jx, config.jy, config.jz, nxglobal, jng);

//                push_particles.update_field(field);
                //Push Particles
                particle_start = timer.seconds();
                Cabana::simd_parallel_for(simd_policy, push_particles, "push_particles");
                Kokkos::fence();
                //Call bcs
                Cabana::simd_parallel_for( simd_policy, pbf, "part_boundary_cond");
                Kokkos::fence();
                particle_end = timer.seconds();
                particle_push_time += (particle_end - particle_start);
                {
                   //Migration
                   //Cabana::Distributor<DeviceType> distributor( MPI_COMM_WORLD, rank_slice, neighbors);
                   //Cabana::migrate( distributor, particle_aosoa );
                   //exchange_data<decltype(particle_aosoa)>( particle_aosoa, neighbors, myrank, particle_aosoa.size());
                   if(nranks > 1){
                    migrator.exchange_data( particle_aosoa, neighbors, myrank, particle_aosoa.size());
                   }
//                   MPI_Allreduce(MPI_IN_PLACE, config.jx.data(), nxglobal, MPI_DOUBLE, MPI_SUM, MPI_COMM_WORLD);
//                   MPI_Allreduce(MPI_IN_PLACE, config.jy.data(), nxglobal, MPI_DOUBLE, MPI_SUM, MPI_COMM_WORLD);
//                   MPI_Allreduce(MPI_IN_PLACE, config.jz.data(), nxglobal, MPI_DOUBLE, MPI_SUM, MPI_COMM_WORLD);
                }
                field_solve_start = timer.seconds();
                current_finish(config.jx, config.jy, config.jz, nxlocal, jng);
                //current_finish(config.jx, config.jy, config.jz, nxglobal, jng);
                field_solve_end = timer.seconds();
                field_solve_time += (field_solve_end - field_solve_start);
            }
           //  check_for_stop_condition()TODO
           if(halt) break;
           step = step + 1;
           time = time + dt / 2.0;
           if(time > next_dump){
                //Deep copy to host fields
                Kokkos::deep_copy(host_ex, config.ex);
                Kokkos::deep_copy(host_ey, config.ey);
                Kokkos::deep_copy(host_ez, config.ez);
                Kokkos::deep_copy(host_bx, config.bx);
                Kokkos::deep_copy(host_by, config.by);
                Kokkos::deep_copy(host_bz, config.bz);
                Kokkos::deep_copy(host_jx, config.jx);
                Kokkos::deep_copy(host_jy, config.jy);
                host_particle_aosoa.resize(particle_aosoa.size());
                Cabana::deep_copy(host_particle_aosoa, particle_aosoa);
                output_routines(host_particle_aosoa, host_ex, host_ey, host_ez,
                                host_bx, host_by, host_bz, host_jx, host_jy, host_jz,
                                dump_count, nxlocal, ng, jng);
               std::cout << "Dump " << dump_count << " after " << timer.seconds() << " seconds.\n";
               //TODO output_routines
               next_dump += per_dump;
               dump_count++;
           }
           time = time + dt / 2.0;
           field_solve_start = timer.seconds();
           update_eb_fields_final(config.ex, config.ey, config.ez, nxglobal, config.jx, config.jy, config.jz, 
                                  config.bx, config.by, config.bz, dt, dx, host_field, config.field, ng,
                                  update_e_field, update_b_field, rangepolicy);
           field_solve_end = timer.seconds();
           field_solve_time += (field_solve_end - field_solve_start);
//           std::cout << "step " << step << ": " << jy(241) << ", " << jz(241) << ", " << by(241) << ", " << ey(241) << ", " << ez(241) << "\n";
        }

    std::cout << "Finishing after " << timer.seconds() << " seconds.\n";
    std::cout << "Field solver time is " << field_solve_time << " seconds.\n";
    std::cout << "Particle push time is " << particle_push_time << " seconds.\n";
    }//End of Kokkos region
    //Kokkos::finalize();
    MPI_Finalize();
    return 0;
}
