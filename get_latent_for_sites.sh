#!/bin/bash
export EP=20
export CUT_START=4
export LINES=288
export SUBJECT=95
export SEQ_LEN=9
export EB=32
python train.py -ep ${EP} --site 'UM' --cut_start ${CUT_START} --lines ${LINES} --subject ${SUBJECT} --seq_len ${SEQ_LEN} -eb ${EB}
export CUT_START=1
export LINES=175
export SUBJECT=173
export SEQ_LEN=7
export EB=25
python train.py -ep ${EP} --site 'NYU' --cut_start ${CUT_START} --lines ${LINES} --subject ${SUBJECT} --seq_len ${SEQ_LEN} -eb ${EB}
export CUT_START=2
export LINES=232
export SUBJECT=70
export SEQ_LEN=8
export EB=29
python train.py -ep ${EP} --site 'USM' --cut_start ${CUT_START} --lines ${LINES} --subject ${SUBJECT} --seq_len ${SEQ_LEN} -eb ${EB}
export CUT_START=1
export LINES=112
export SUBJECT=71
export SEQ_LEN=7
export EB=16
python train.py -ep ${EP} --site 'UCLA' --cut_start ${CUT_START} --lines ${LINES} --subject ${SUBJECT} --seq_len ${SEQ_LEN} -eb ${EB}