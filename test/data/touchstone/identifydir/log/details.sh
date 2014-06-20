## segway (%[^ ]+%) run (%[0-9a-f]{32}%) at (%[0-9]{4}%)-(%[0-9]{2}%)-(%[0-9]{2}%) (%[0-9]{2}%):(%[0-9]{2}%):(%[0-9]{2}%).(%[0-9]{1,}%)
(%[^ ]+%)/gmtkTriangulate -cppCommandOptions "-DCARD_SEG=4 -DCARD_FRAMEINDEX=2000000 -DSEGTRANSITION_WEIGHT_SCALE=1.0 -DCARD_SUBSEG=1" -outputTriangulatedFile identifydir/triangulation/segway.str.4.1.trifile -strFile traindir/segway.str -verbosity 0
(%[^ ]+%)/segway-task run viterbi identifydir/viterbi/viterbi0.bed chr21 9411193 9595548 1 0 4 seg ../test.genomedata (%[^ ]+%)/chr21.0000.(%[0-9a-f]{32}%).float32 (%[^ ]+%)/chr21.0000.(%[0-9a-f]{32}%).int asinh_norm 0,1 -base 3 -cVitRegexFilter ^[a-z]*seg$ -cliqueTableNormalize 0.0 -componentCache F -cppCommandOptions "-DCARD_SEG=4 -DINPUT_PARAMS_FILENAME=traindir/params/params.params -DCARD_FRAMEINDEX=2000000 -DSEGTRANSITION_WEIGHT_SCALE=1.0 -DCARD_SUBSEG=1" -deterministicChildrenStore F -eVitRegexFilter ^[a-z]*seg$ -fmt1 binary -fmt2 binary -hashLoadFactor 0.98 -inputMasterFile traindir/params/input.master -island T -iswp1 F -iswp2 F -jtFile identifydir/log/jt_info.txt -lst 100000 -mVitValsFile - -nf1 2 -nf2 0 -ni1 0 -ni2 2 -obsNAN T -of1 identifydir/observations/float32.list -of2 identifydir/observations/int.list -pVitRegexFilter ^[a-z]*seg$ -strFile traindir/segway.str -triFile identifydir/triangulation/segway.str.4.1.trifile -verbosity 0 -vitCaseSensitiveRegexFilter T
(%[^ ]+%)/segway-task run posterior identifydir/posterior/posterior%s.0.bed chr21 9411193 9595548 1 0 4 seg ../test.genomedata (%[^ ]+%)/chr21.0000.(%[0-9a-f]{32}%).float32 (%[^ ]+%)/chr21.0000.(%[0-9a-f]{32}%).int asinh_norm 0,1 -base 3 -cCliquePrintRange 1:1 -cliqueTableNormalize 0.0 -componentCache F -cppCommandOptions "-DCARD_SEG=4 -DINPUT_PARAMS_FILENAME=traindir/params/params.params -DCARD_FRAMEINDEX=2000000 -DSEGTRANSITION_WEIGHT_SCALE=1.0 -DCARD_SUBSEG=1" -deterministicChildrenStore F -doDistributeEvidence T -eCliquePrintRange 1:1 -fmt1 binary -fmt2 binary -hashLoadFactor 0.98 -inputMasterFile traindir/params/input.master -island T -iswp1 F -iswp2 F -jtFile identifydir/log/jt_info.posterior.txt -lst 100000 -nf1 2 -nf2 0 -ni1 0 -ni2 2 -obsNAN T -of1 identifydir/observations/float32.list -of2 identifydir/observations/int.list -pCliquePrintRange 1:1 -strFile traindir/segway.str -triFile identifydir/triangulation/segway.str.4.1.posterior.trifile -verbosity 0

