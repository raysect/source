# Copyright (c) 2014-2023, Dr Alex Meakins, Raysect Project
# All rights reserved.
#
# Redistribution and use in source and binary forms, with or without
# modification, are permitted provided that the following conditions are met:
#
#     1. Redistributions of source code must retain the above copyright notice,
#        this list of conditions and the following disclaimer.
#
#     2. Redistributions in binary form must reproduce the above copyright
#        notice, this list of conditions and the following disclaimer in the
#        documentation and/or other materials provided with the distribution.
#
#     3. Neither the name of the Raysect Project nor the names of its
#        contributors may be used to endorse or promote products derived from
#        this software without specific prior written permission.
#
# THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS"
# AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
# IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE
# ARE DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT HOLDER OR CONTRIBUTORS BE
# LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR
# CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF
# SUBSTITUTE GOODS OR SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS
# INTERRUPTION) HOWEVER CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN
# CONTRACT, STRICT LIABILITY, OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE)
# ARISING IN ANY WAY OUT OF THE USE OF THIS SOFTWARE, EVEN IF ADVISED OF THE
# POSSIBILITY OF SUCH DAMAGE.

"""
Unit tests for the core functions of the pseudo random number generator.
"""

import unittest
from raysect.core.math.random import seed, uniform

# generated with seed(1234567890)
_random_reference = [
    0.8114659955555504, 0.0260575258293142, 0.21664518027139346, 0.036431793406025315, 0.34039768629173206,
    0.5307847417000392, 0.16688396521453341, 0.9019057801125824, 0.8159473905407517, 0.8570265343145624,
    0.14347871551095648, 0.6891296327000385, 0.5778067096794773, 0.33066470187437946, 0.6876265206072084,
    0.886939227534345, 0.014324369430429362, 0.6746414734836368, 0.7405781756866624, 0.331961408614173,
    0.290705874656528, 0.9380405182274753, 0.2709760824131112, 0.31300988656894224, 0.3103875692954393,
    0.6280866311578346, 0.2994918127081234, 0.05769585538469579, 0.009078614743584623, 0.8087765866312592,
    0.4470374188892262, 0.06707140588005311, 0.5503547474504604, 0.9431449802430566, 0.2588098326024648,
    0.4320869116583066, 0.5995417081018991, 0.013438112912753097, 0.8728231064672088, 0.878780254056249,
    0.36288545060141253, 0.27384446607131097, 0.6874217254488153, 0.17265752434898118, 0.19655596596529323,
    0.9146957191169391, 0.7740513037040511, 0.23969347216349357, 0.3085577835996567, 0.5750617634056108,
    0.7607788813988423, 0.5461998954715511, 0.07226744025005805, 0.30272312199484996, 0.07240473306707063,
    0.7445783003493487, 0.023204172936409417, 0.3541488588854369, 0.288366130356135, 0.9896030024662023,
    0.33897243830956825, 0.19899954974241374, 0.12867372424139678, 0.19636195456378958, 0.3864429701036983,
    0.3077401387398464, 0.5799414672883465, 0.43022673663871036, 0.5201589558513544, 0.8257246242158377,
    0.7866170888100589, 0.20812523470942856, 0.3580046317178087, 0.8209043595111258, 0.6016131136683351,
    0.9215493153953329, 0.4532258113173726, 0.8986509588916062, 0.8333655212557587, 0.1660658459237977,
    0.11269071968496114, 0.6231919445394714, 0.04035882401424995, 0.0563724857548058, 0.19756368026802662,
    0.1965944584402416, 0.415015687836517, 0.027876686089227776, 0.9996656197629226, 0.4175304119184161,
    0.06710393664544856, 0.8922085748965133, 0.028869646904276958, 0.5788803807634426, 0.9682232374298705,
    0.6163314615766219, 0.2830323089125376, 0.5029784234700792, 0.8420338434150371, 0.4651270080273381,
    0.6557994530936825, 0.3292144780285112, 0.3225072195563581, 0.7399474648179799, 0.48079437548223924,
    0.7199262095106047, 0.5975905519484589, 0.9236696093086989, 0.7346397307479617, 0.3592223034718819,
    0.561867086066638, 0.03995876009164412, 0.49890487301779407, 0.44713823160917965, 0.7861240452195936,
    0.25911075097295855, 0.035083102638025165, 0.36607136391410944, 0.19984528149815695, 0.9500543496932649,
    0.23731801886771908, 0.6006092389019372, 0.8294417196340962, 0.052764856825375794, 0.6038433712269217,
    0.7949850355716386, 0.6605823678792535, 0.8189544045861851, 0.022260100040748965, 0.576004798924091,
    0.9268911541444015, 0.4896501699585003, 0.8873523166370707, 0.03815088769851427, 0.9776353573173021,
    0.5145565026412993, 0.32552527078939697, 0.4911956494196167, 0.8951990538514405, 0.7885065235998865,
    0.6625825765906797, 0.3198923578390026, 0.4894245309430685, 0.8514988780345353, 0.31964651050694937,
    0.34732693542699866, 0.6386792428329865, 0.9352495191889876, 0.46790614266368125, 0.011411382010219517,
    0.9775139174995728, 0.3862808053921021, 0.8529921244254951, 0.2605262061279382, 0.6922576403464112,
    0.15045572118688688, 0.41161431623563194, 0.7126021992146101, 0.4171153209324736, 0.110043850688747,
    0.17730501058976444, 0.25657426179278153, 0.2821384093255612, 0.8597361850849774, 0.3935528040200792,
    0.8949494271905098, 0.5441559448691042, 0.27510096681653573, 0.5223954176767841, 0.17454850986131998,
    0.8883671592862563, 0.31112566318232215, 0.5560927365286331, 0.14008244909651368, 0.009760829072095967,
    0.8525804007596959, 0.6921346351519599, 0.38891810962766127, 0.17941655286803415, 0.5904814324711654,
    0.09079027074688151, 0.0942749621536777, 0.10551209561705677, 0.3409843480241793, 0.11546679641182545,
    0.5460531038179809, 0.7292181368445171, 0.9041051293305149, 0.07400110842637242, 0.8569642064071208,
    0.2582827697271819, 0.17032689124026934, 0.7235634272987398, 0.899745025204368, 0.5215209486597834,
    0.6991051509833436, 0.5183070756019655, 0.8666000508220529, 0.057923176060497794, 0.5062702174339315,
    0.8723895729920971, 0.6948914830488712, 0.6489433851279749, 0.5306528480010244, 0.8225027378406932,
    0.7838244572343968, 0.691544103178456, 0.7831551000165751, 0.6742553578920283, 0.28556616290828807,
    0.9441828882154812, 0.2897573727609368, 0.27092040074004464, 0.5221435897948234, 0.5147483528375717,
    0.4434962915318513, 0.5514590705654433, 0.49042130275752494, 0.3090263245842645, 0.05335198870897018,
    0.303566210158679, 0.39691290229966336, 0.6565826934102237, 0.6843076816775873, 0.9390799941122242,
    0.4565438740955222, 0.5834686885918711, 0.6345292041511743, 0.03435573177180418, 0.8570048259135065,
    0.9945422765096777, 0.33539171490048414, 0.9234474376753186, 0.36969772769699494, 0.14248787800080065,
    0.060836330283883466, 0.6876555335527527, 0.2524343868154566, 0.7003003061613287, 0.5423027600372561,
    0.8142644438311161, 0.16459341480713097, 0.5896280051306514, 0.9100813849651408, 0.8464665169207302,
    0.29377649889194857, 0.4917633089616589, 0.8615633378217579, 0.940272146073147, 0.08225016215748171,
    0.8914723950338176, 0.7485079492976437, 0.16549658948390567, 0.5315626289230118, 0.9450558307395565,
    0.044836457646848404, 0.6589484995600837, 0.6639350986521546, 0.40094076346047536, 0.621552238087893,
    0.8463783046116087, 0.5279429414308022, 0.8733012361951688, 0.3753618518585796, 0.6199087760073151,
    0.6638980416420605, 0.9519166805682214, 0.7714573635177544, 0.532306948913898, 0.9950699313841207,
    0.6023812221802153, 0.00671953993578811, 0.41257363188340057, 0.22374577891859204, 0.9587200148497917,
    0.5656973690886793, 0.12295446157634149, 0.14384586782995712, 0.2999263339675363, 0.19016587334152224,
    0.7371161197609226, 0.963678596691637, 0.19459831960744067, 0.023732861037621178, 0.9022346289689608,
    0.817551073855459, 0.8565204343428555, 0.34083263803209185, 0.3147716178439317, 0.22924021573146747,
    0.006945465955912389, 0.34414493802070234, 0.023818385299195555, 0.035214663181623806, 0.6928101211795044,
    0.05857375424125044, 0.06501678645231457, 0.7381578101112583, 0.8018121479956365, 0.4426125308472515,
    0.5334659795706291, 0.241504539949834, 0.34846402489916717, 0.3160766708257674, 0.6820437033453958,
    0.48572662123104837, 0.2276401005217804, 0.8245514413657292, 0.5519253690402882, 0.251912099236969,
    0.40793686767585957, 0.7151296734180584, 0.9824217073234381, 0.11529480920795265, 0.8723676328924153,
    0.57771290763045, 0.871023854555776, 0.24685776479884225, 0.22934957742352136, 0.6002908335180984,
    0.8952992211384087, 0.3713611310051842, 0.56407375667193, 0.5953393802057543, 0.29828973376345236,
    0.11966741726342045, 0.6987001221692178, 0.7878557907165742, 0.06390695747815511, 0.08355291052833858,
    0.7425821939621605, 0.9566467958396809, 0.1342316008404204, 0.33043778095310317, 0.5961385752911869,
    0.7162577188447022, 0.41151980274036737, 0.6287447927091593, 0.9000573601374122, 0.8857847936198314,
    0.6618915691643112, 0.10901537670332251, 0.974332847977026, 0.7430921207569577, 0.3738798578677184,
    0.3177953674650934, 0.17907762643786318, 0.6223416913159243, 0.41761298976200534, 0.33313570677119575,
    0.7364457497838391, 0.27596382366048255, 0.6424208011354875, 0.7057512390482654, 0.43406222148090023,
    0.9713038931369188, 0.016579486340549843, 0.8559782534797721, 0.8285887019972442, 0.45041719041568773,
    0.8385573140413476, 0.3598494489892735, 0.010566283696215928, 0.7919540721893159, 0.7572641485392072,
    0.9828556345311235, 0.44749216530159, 0.775872823653567, 0.21441646102494516, 0.5469506800802856,
    0.6250315474032357, 0.7228454114709449, 0.7762777922445003, 0.3725320514474523, 0.1997830988186624,
    0.8422680077727865, 0.5174000366043017, 0.4114337679642368, 0.8485359821345219, 0.8105411087109267,
    0.14698078060577857, 0.5140303811686008, 0.7243343817393784, 0.4550925694099044, 0.3202637711960984,
    0.8189363549218315, 0.7566860383375725, 0.7642137638017872, 0.7118547919654123, 0.25868958239576423,
    0.26071980348118695, 0.0014416717989212957, 0.2531116783475601, 0.7452981354849321, 0.22666538044066054,
    0.38860664857167593, 0.5727684813549001, 0.34391041728313887, 0.10125872981993811, 0.3698391553828596,
    0.37392763598955614, 0.9352150532839864, 0.3196533316770499, 0.9451329545264995, 0.7011577054305139,
    0.4972479758125282, 0.19394292978600158, 0.2255784709723747, 0.8497665546392211, 0.7784459333176916,
    0.3826938886585244, 0.8823373252575396, 0.5095613352700986, 0.07043189724308563, 0.2651448359964339,
    0.7756594236158155, 0.9333994810847233, 0.8625965208644846, 0.27085634631115396, 0.544146314956061,
    0.005441946968595568, 0.1725375800157205, 0.8119268223977082, 0.41820424887848173, 0.40334273244470054,
    0.14905838003830352, 0.39670562066321646, 0.4526807368029184, 0.9690226266124907, 0.15634095136705828,
    0.26766468856658987, 0.18338178075941547, 0.992322802253603, 0.5236539345496551, 0.2985956624859313,
    0.4100434474858442, 0.30853502687943635, 0.0887764814470422, 0.9796372081229322, 0.24109222934272623,
    0.5050019585901817, 0.9454834743694469, 0.2521111838931931, 0.30393236334610707, 0.7590426698546433,
    0.32565502384735956, 0.3618088380165515, 0.7603049752109582, 0.8079375811328675, 0.9640427963149142,
    0.09107650871535244, 0.7053020025232072, 0.20134358439883027, 0.0969959330330612, 0.7634817795040539,
    0.14288160000682015, 0.8995689280060272, 0.4121730916081331, 0.6381011014457788, 0.5347500291497603,
    0.6981388766633505, 0.6086556203191144, 0.0997947219673625, 0.5180451101421524, 0.8457865844959731,
    0.4802718326400941, 0.5480160414929314, 0.6692537196809561, 0.6086772632135882, 0.5755883318728027,
    0.8928160726545578, 0.4537771248497059, 0.5855826059810336, 0.9693649946390455, 0.9809955907195725,
    0.8222605121455328, 0.9315157929561733, 0.7610683650907902, 0.18453699065691143, 0.18987653993548925,
    0.08058285379914143, 0.13258640947297373, 0.9276014353057904, 0.6826008435990373, 0.5827558781355116,
    0.672636560507286, 0.6961240608292445, 0.16644178433862533, 0.21423970488961208, 0.547607323698878,
    0.8410374079396743, 0.13582173342038273, 0.9112486015153621, 0.8940761424722561, 0.041135346871277734,
    0.8930613830991306, 0.2897163426842069, 0.7093705018045361, 0.3981338285552244, 0.7910226772822321,
    0.8889762775630399, 0.3790304426009268, 0.8601770030681934, 0.6138027901369926, 0.7056320576164431,
    0.2476095758742285, 0.8132910616402856, 0.8689430742973778, 0.9295565490379185, 0.2841332486425633,
    0.7761638838352205, 0.2628043693331882, 0.39488597717393137, 0.10902072616095637, 0.32922801464636264,
    0.9893075329888635, 0.6042356789402263, 0.7257762472267762, 0.3304761279554156, 0.16014540968416557,
    0.7244656911898267, 0.28200634660998836, 0.1955121568345718, 0.27844276258787903, 0.27617018599531706,
    0.12640476624099528, 0.4975971208046923, 0.03530997699866356, 0.5588208829600507, 0.6309320467769252,
    0.9927038509048303, 0.43155203248849583, 0.8743471515148651, 0.014292646197526904, 0.6348358539197627,
    0.2088903476691828, 0.7400772190991796, 0.6118159913938324, 0.8768029651595749, 0.8965458010407473,
    0.8550853428733064, 0.7895834058595897, 0.6787044063820584, 0.3033069747750138, 0.563032540187197,
    0.2866958456729516, 0.6804805688846556, 0.28086923543329423, 0.7090636048238402, 0.7834965258386127,
    0.4512887050774034, 0.355423710061168, 0.08428225259227184, 0.4127942317900083, 0.10136925268583408,
    0.3718144648996242, 0.021269730316966662, 0.1472541613547963, 0.5542936329158991, 0.6443057701942396,
    0.3807351756123216, 0.6072488055686531, 0.37037940366328637, 0.8319233029191506, 0.7550907359873181,
    0.043204485029981865, 0.2057526694309575, 0.5610640391262, 0.6164096646792445, 0.5561287744052342,
    0.42914845719986494, 0.5964488034004436, 0.8527268944513903, 0.8270737747453121, 0.2623987600454559,
    0.8922733979408876, 0.21408558359169705, 0.7769381051002117, 0.1382750585458662, 0.6050849827198906,
    0.21766895125825292, 0.6067790813241967, 0.8082650866597821, 0.34813166222037506, 0.4165288897956795,
    0.8210636441113087, 0.609646597542757, 0.5372994025479527, 0.4849626034523542, 0.41447857298142254,
    0.8206055713161226, 0.3375955171501559, 0.5133358994389254, 0.7830192378510232, 0.4407111355244818,
    0.3995487295266026, 0.22822716386734787, 0.32860765082307486, 0.33763957464687633, 0.054434622315739256,
    0.8094181305493572, 0.535453838731794, 0.9600813314329106, 0.8590236593122432, 0.022493420031869715,
    0.30705758228596525, 0.01563195672690343, 0.3174866967434624, 0.772092899011473, 0.7054066010354223,
    0.048859586900376506, 0.4768127919797456, 0.26256897983199623, 0.3889740130240964, 0.677986412352965,
    0.2598445428360996, 0.6496040121038401, 0.8103854067681818, 0.9800399860695725, 0.9055670105357904,
    0.7097953273255118, 0.3585186662150981, 0.5568353686574206, 0.08845545202125726, 0.16072563815750252,
    0.20508470659388278, 0.5085201598548588, 0.21301485459742509, 0.8342799458856295, 0.03142978077813019,
    0.31157069077044186, 0.9235515901806688, 0.4669799028719711, 0.18642602380883333, 0.6431751681934158,
    0.6531267612980451, 0.7699808254096996, 0.07559532762088994, 0.08574910096987087, 0.4725201993249537,
    0.4900671456656317, 0.0824182639396196, 0.5348469916235775, 0.03945363362033638, 0.40093997185570807,
    0.46480434300517903, 0.6308751528484047, 0.41724094123184474, 0.09877249492678075, 0.9088970559325534,
    0.781519477292224, 0.16620566965553463, 0.7128625312605764, 0.06687693207244172, 0.04209704414327786,
    0.1162134199127226, 0.28098456619669065, 0.4146409837206263, 0.9009174193057882, 0.5647737811880079,
    0.7818660970751818, 0.4688238338261841, 0.14677089233924467, 0.07791518888221172, 0.5318220598787906,
    0.5700281022868329, 0.4028158916854355, 0.7152930324528218, 0.8445061967460851, 0.29118812322506016,
    0.9004598751238156, 0.11220581130720386, 0.5290991553810941, 0.7907499464570443, 0.11997903435659885,
    0.1092731941756504, 0.9861516232513136, 0.9641493160738513, 0.45131753627208837, 0.2534491566423186,
    0.8035717495221812, 0.685198258275126, 0.5278965268586229, 0.3854252546562611, 0.03314184206146109,
    0.039607736084224854, 0.6278111677701048, 0.6152808786614328, 0.13966613726902255, 0.8620583580194112,
    0.9669361765861261, 0.764603363709067, 0.32162131735620114, 0.32739250166838896, 0.8048913301525041,
    0.09013040808534978, 0.6780262054946375, 0.14955962962919134, 0.7014380395536837, 0.10703430614218357,
    0.2725502659791028, 0.6109287614522758, 0.9701216262369222, 0.7077671112634293, 0.28262240295934615,
    0.7429869273594056, 0.30343910138584196, 0.4387989074763934, 0.685560617408065, 0.9212628522979792,
    0.6710972049337736, 0.6678957086131286, 0.052485043752963256, 0.03262215173872851, 0.5533794367051991,
    0.14334883173534696, 0.8750261712197275, 0.21849295019688597, 0.789900305335936, 0.04378837917322642,
    0.8448111409516583, 0.8554817655418667, 0.5422899883799648, 0.8322998447663251, 0.9043309648830229,
    0.11376878252583256, 0.010574450853024775, 0.6059205512202163, 0.05650192227932216, 0.06616182593422049,
    0.300840412056413, 0.11009113618103572, 0.09017278454547817, 0.7324074703533104, 0.5887507946661643,
    0.8011993764162316, 0.9537044413248185, 0.3480848466091717, 0.04492720945080131, 0.12772985315204488,
    0.09318647234488864, 0.4177564546636331, 0.17908052958405896, 0.9377559994221184, 0.7942001034678936,
    0.018264818293493024, 0.09929932833940613, 0.9816497277698718, 0.7254971313087528, 0.5697799313851291,
    0.20738382816360446, 0.9413770950513797, 0.19089363309518015, 0.7246898485586507, 0.22277616586322624,
    0.2756137473006507, 0.5806902069622485, 0.15121601677076346, 0.26219984984480804, 0.5617317198216203,
    0.5340793495810867, 0.7802656945952358, 0.2563728865278596, 0.9722632323000315, 0.12678239231988597,
    0.5926580329225439, 0.9471136072043423, 0.7929900300971447, 0.1671869820758628, 0.3971575496869405,
    0.4091510156827586, 0.7779342209221878, 0.15077561988451882, 0.24865991522931574, 0.7577232053268574,
    0.8553874230783778, 0.39172177799779173, 0.6278466931159832, 0.5846812724737959, 0.6457445565848937,
    0.6267819035915743, 0.9094479540235458, 0.5693214474531362, 0.21078166015345667, 0.2301828826215616,
    0.4192657341771445, 0.548538931771065, 0.7361467529623559, 0.3789302500804076, 0.676427374762233,
    0.3313045125291517, 0.06780277293961923, 0.5778351563517666, 0.8911852199571377, 0.5728970528544474,
    0.9699817028914329, 0.43039129706514545, 0.08674775688446801, 0.3675919508116977, 0.9225119580037218,
    0.6988194440402635, 0.5253181253502084, 0.6456804731969746, 0.26339182772715775, 0.22626993807075557,
    0.550356185891607, 0.56814132057691, 0.23140177254779548, 0.08163330108694333, 0.6950026943794549,
    0.3868874524520275, 0.7803560695948997, 0.8111106020547105, 0.334403887879808, 0.07747054777469231,
    0.131679890515222, 0.385738851847096, 0.5720750915742275, 0.6537300415909767, 0.033579514204037375,
    0.6267871762497315, 0.283315936418973, 0.6968535140130128, 0.32523324654122043, 0.34672320660673406,
    0.27957808136314133, 0.5658693620920029, 0.7989747162309463, 0.6135521139363683, 0.35171172850223953,
    0.6840952845832683, 0.7617108586929001, 0.08385365215958673, 0.8377444841629399, 0.4058061560932523,
    0.8520764020443331, 0.22112770192078446, 0.3597174340725352, 0.7681341972913422, 0.4489397921948374,
    0.7690271682513512, 0.9343709661134892, 0.7537016686581054, 0.8532417940641944, 0.21786544321625967,
    0.9220532297103946, 0.6044137130573399, 0.8477580054395553, 0.3650904814611189, 0.550454839779165,
    0.35397013127490295, 0.3167770033683245, 0.6019910259733843, 0.9006467399545588, 0.966868800513041,
    0.8287187605620165, 0.34122925546186533, 0.4656526174427663, 0.43334812328455674, 0.013263490405688083,
    0.26612364947878253, 0.5131104482037588, 0.8864329675456142, 0.9017844237441688, 0.46913085233212526,
    0.9565194929638963, 0.48331540015439556, 0.8942705810265009, 0.7576375600904258, 0.5523909581952984,
    0.43295148471017175, 0.8512591519488544, 0.6833666101748067, 0.4888591152858486, 0.6402955434059505,
    0.6208145089126011, 0.2477338750104311, 0.21603797561049176, 0.8150167527352777, 0.9726705712342355,
    0.3214682713106668, 0.8624495481477454, 0.6930029337014305, 0.4749272194303096, 0.8258052149906059,
    0.6687961480135753, 0.8120600596698068, 0.15420597835671246, 0.5782577479558763, 0.5807398032680138,
    0.978331447998591, 0.9396038337478894, 0.6913963473451303, 0.9188691280946835, 0.9530423584052496,
    0.8676848853851221, 0.8158943909093959, 0.10737951927744527, 0.47262298717511697, 0.10378819638161618,
    0.9276478971005653, 0.6428852905573831, 0.17746103075445874, 0.3020170597330829, 0.22401044731747322,
    0.0577426005330417, 0.8985902135238854, 0.9615034506483714, 0.9787531208633925, 0.6234050362666279,
    0.10818578608950069, 0.08002597075571494, 0.40612610122984627, 0.5557805970069319, 0.972064174913354,
    0.6279093472467558, 0.2558826735265932, 0.04606558330313981, 0.00919631584008862, 0.09046938247961245,
    0.6794105465566141, 0.60732321072077, 0.2541301129844721, 0.3408459253369911, 0.7521255057949666,
    0.6254903026485684, 0.9890982642442152, 0.7086637519333243, 0.3260301174978997, 0.9132255776141602,
    0.5266524126000847, 0.12872091545378628, 0.5173647446744717, 0.040958411123854344, 0.6212975171513406,
    0.43502437401513183, 0.42646995770490204, 0.2692764271641708, 0.6680813915009773, 0.6767636171224242,
    0.5000158219375631, 0.8255995563326667, 0.08966422549525332, 0.5632935338808306, 0.14705474560748233,
    0.40403318221002793, 0.14701374362900166, 0.0834293465054069, 0.7314121992911785, 0.27889710619969366,
    0.05459710747091584, 0.48311405862113443, 0.36523989425663217, 0.5858606795260355, 0.17073141155483396,
    0.06466140598006276, 0.7782744447593027, 0.45005338754651747, 0.6405860800740856, 0.2305203445449855,
    0.6501759636740025, 0.33332034091907814, 0.5843937688287243, 0.5586369497088509, 0.9789078545145837,
    0.9163241892373197, 0.6008963135873424, 0.7032170412450808, 0.22511115467783138, 0.2687027228917952,
    0.6641315333336573, 0.6227339938112236, 0.20518174508232712, 0.13853586263077233, 0.005866161059393082,
    0.9578758962894522, 0.6605760502889816, 0.8246940071965811, 0.20054547619996677, 0.8120300891449976,
    0.08925190901053293, 0.5157538258011378, 0.008446965357773117, 0.8937482962690314, 0.9542808129039703,
    0.041094266919463385, 0.22961677645946077, 0.5941572693245096, 0.21538079934488563, 0.9297485411139692,
    0.7313599834684, 0.8514206012515507, 0.37969815204001134, 0.5956592217572321, 0.12323732942423993,
    0.8453301040453715, 0.8127586375364623, 0.9237285841574706, 0.7996900004382563, 0.7120644121842296,
]


class TestRandom(unittest.TestCase):

    def test_random(self):
        """
        Tests the pseudo random number generator.
        """

        seed(1234567890)
        for v in _random_reference:
            self.assertEqual(uniform(), v, msg="Random failed to reproduce the reference data.")


