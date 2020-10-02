/* eslint-disable max-len */

module.exports = {
  took: 2,
  timed_out: false,
  _shards: {
    total: 1,
    successful: 1,
    skipped: 0,
    failed: 0,
  },
  hits: {
    total: 200,
    max_score: null,
    hits: [
      {
        _index: 'c490fa3a0fb8fdfe42586242ae9ae3b8',
        _type: 'docs',
        _id: '--rB52YBhEpkl2gfITkY',
        _score: 1.0,
        _source: {
          input: {
            tokens:
              'Tigerland is one of the finest films that i have seen, and in my opinion it outdoes even full metal jacket, a film of similar nature. Bozz is played exceptionally well by Farrell, and is a character who stays in your mind long after the film ends. The ending is brilliantly cut by schumacher - with the melodic harmony singing and the slow mo of the troops preparing to leave. What a film.',
            gold_label: 'positive',
            '@source': {
              label: 'positive',
              text:
                'Tigerland is one of the finest films that i have seen, and in my opinion it outdoes even full metal jacket, a film of similar nature. Bozz is played exceptionally well by Farrell, and is a character who stays in your mind long after the film ends. The ending is brilliantly cut by schumacher - with the melodic harmony singing and the slow mo of the troops preparing to leave. What a film.',
            },
          },
          prediction: {
            logits: [-0.8541305661201477, 0.8605541586875916],
            classes: {
              negative: 0.15255707502365112,
              positive: 0.8474429249763489,
            },
            max_class: 'positive',
            max_class_prob: 0.8474429249763489,
          },
          feedback: {
            prediction: {
              max_class: 'negative',
              max_class_prob: 1,
            },
            status: 'rejected',
          },
        },
        sort: [1.0, '--rB52YBhEpkl2gfITkY'],
      },
      {
        _index: 'c490fa3a0fb8fdfe42586242ae9ae3b8',
        _type: 'docs',
        _id: '-OrB52YBhEpkl2gfITkY',
        _score: 1.0,
        _source: {
          input: {
            tokens:
              'Tears of Kali is an original yet flawed horror film that delves into the doings of a cult group in India comprised of German psychologists who have learned how to control their wills and their bodies to the point that they can cause others to be \'healed\' through radical techniques (that can trigger nightmarish hallucinations and physical pain and torture) to release the pent-up demons inside them.<br /><br />The film is shown as a series of vignettes about the Taylor-Eriksson group--the above-mentioned cult group. The first segment is somewhat slower than the rest but serves fine to set up the premise for the rest of the film. The rest of it plays out like a mindf@ck film with some of the key staples thrown in the mix (full-frontal nudity, some gore) to keep you happy.<br /><br />I say check this out. May not be spectacular, but it\'s concept is pretty neato and it delivers in the right spots. 8/10.',
            gold_label: 'positive',
            '@source': {
              label: 'positive',
              text:
                'Tears of Kali is an original yet flawed horror film that delves into the doings of a cult group in India comprised of German psychologists who have learned how to control their wills and their bodies to the point that they can cause others to be \'healed\' through radical techniques (that can trigger nightmarish hallucinations and physical pain and torture) to release the pent-up demons inside them.<br /><br />The film is shown as a series of vignettes about the Taylor-Eriksson group--the above-mentioned cult group. The first segment is somewhat slower than the rest but serves fine to set up the premise for the rest of the film. The rest of it plays out like a mindf@ck film with some of the key staples thrown in the mix (full-frontal nudity, some gore) to keep you happy.<br /><br />I say check this out. May not be spectacular, but it\'s concept is pretty neato and it delivers in the right spots. 8/10.',
            },
          },
          prediction: {
            logits: [-1.1640548706054688, 1.3028548955917358],
            classes: {
              negative: 0.07821071892976761,
              positive: 0.921789288520813,
            },
            max_class: 'positive',
            max_class_prob: 0.921789288520813,
          },
          feedback: {
            prediction: {
              max_class: 'negative',
              max_class_prob: 1,
            },
            status: 'rejected',
          },
        },
        sort: [1.0, '-OrB52YBhEpkl2gfITkY'],
      },
      {
        _index: 'c490fa3a0fb8fdfe42586242ae9ae3b8',
        _type: 'docs',
        _id: '-erB52YBhEpkl2gfITkY',
        _score: 1.0,
        _source: {
          input: {
            tokens:
              'I really liked this movie, and went back to see it two times more within a week.<br /><br />Ms. Detmers nailed the performance - she was like a hungry cat on the prowl, toying with her prey. She lashes out in rage and lust, taking a \'too young\' lover, and crashing hundreds of her terrorist fiancé\'s mother\'s pieces of fine china to the floor. <br /><br />The film was full of beautiful touches. The Maserati, the wonderful wardrobe, the flower boxes along the rooftops. I particularly enjoyed the ancient Greek class and the recitation of \'Antigone\'.<br /><br />It had a feeling of \'Story of O\' - that is, where people of means indulge in unrestrained sexual adventure. As she walks around the fantastic apartment in the buff, she is at ease - and why not, what is to restrain a \'Devil in the Flesh\'?<br /><br />The whole movie is a real treat!',
            gold_label: 'positive',
            '@source': {
              label: 'positive',
              text:
                'I really liked this movie, and went back to see it two times more within a week.<br /><br />Ms. Detmers nailed the performance - she was like a hungry cat on the prowl, toying with her prey. She lashes out in rage and lust, taking a \'too young\' lover, and crashing hundreds of her terrorist fiancé\'s mother\'s pieces of fine china to the floor. <br /><br />The film was full of beautiful touches. The Maserati, the wonderful wardrobe, the flower boxes along the rooftops. I particularly enjoyed the ancient Greek class and the recitation of \'Antigone\'.<br /><br />It had a feeling of \'Story of O\' - that is, where people of means indulge in unrestrained sexual adventure. As she walks around the fantastic apartment in the buff, she is at ease - and why not, what is to restrain a \'Devil in the Flesh\'?<br /><br />The whole movie is a real treat!',
            },
          },
          prediction: {
            logits: [-0.8452581763267517, 0.9247906804084778],
            classes: {
              negative: 0.14553625881671906,
              positive: 0.8544637560844421,
            },
            max_class: 'positive',
            max_class_prob: 0.8544637560844421,
          },
          feedback: {
            prediction: {
              max_class: 'negative',
              max_class_prob: 1,
            },
            status: 'rejected',
          },
        },
        sort: [1.0, '-erB52YBhEpkl2gfITkY'],
      },
      {
        _index: 'c490fa3a0fb8fdfe42586242ae9ae3b8',
        _type: 'docs',
        _id: '-urB52YBhEpkl2gfITkY',
        _score: 1.0,
        _source: {
          input: {
            tokens:
              'Possible spoilers re: late-appearing cameos <br /><br />Seldom does one see so many fine & memorable character actors (almost entirely actresses to be precise) in one film. Even though a few only appear for cameos, each one is a gem. The British do this mix of comedy and real-life pathos better than anyone IMO, so it is no surprise that most of the actors are Brits.<br /><br />The music is great; no doubt much had to be dubbed (does Leslie Caron *really* play the bass so well? maybe - who knew?) But Clio Lane was unmistakably herself - her warm visceral sound still turns my crank like few other jazz singers.<br /><br />As an aging musician myself, not quite as old and certainly not in that class of course, it was a heartening film as well -- a great film for anyone whose wondering if they\'re past it in their profession or avocation whatever it may be. And of course a great celebration of the life of the stage.<br /><br />I missed a little of the opening, but a provisional 9/10 -- and it certainly makes we want to see the whole thing.',
            gold_label: 'positive',
            '@source': {
              label: 'positive',
              text:
                'Possible spoilers re: late-appearing cameos <br /><br />Seldom does one see so many fine & memorable character actors (almost entirely actresses to be precise) in one film. Even though a few only appear for cameos, each one is a gem. The British do this mix of comedy and real-life pathos better than anyone IMO, so it is no surprise that most of the actors are Brits.<br /><br />The music is great; no doubt much had to be dubbed (does Leslie Caron *really* play the bass so well? maybe - who knew?) But Clio Lane was unmistakably herself - her warm visceral sound still turns my crank like few other jazz singers.<br /><br />As an aging musician myself, not quite as old and certainly not in that class of course, it was a heartening film as well -- a great film for anyone whose wondering if they\'re past it in their profession or avocation whatever it may be. And of course a great celebration of the life of the stage.<br /><br />I missed a little of the opening, but a provisional 9/10 -- and it certainly makes we want to see the whole thing.',
            },
          },
          prediction: {
            logits: [-0.549193799495697, 0.6055325865745544],
            classes: {
              negative: 0.23962683975696564,
              positive: 0.7603731155395508,
            },
            max_class: 'positive',
            max_class_prob: 0.7603731155395508,
          },
          feedback: {
            prediction: {
              max_class: 'negative',
              max_class_prob: 1,
            },
            status: 'rejected',
          },
        },
        sort: [1.0, '-urB52YBhEpkl2gfITkY'],
      },
      {
        _index: 'c490fa3a0fb8fdfe42586242ae9ae3b8',
        _type: 'docs',
        _id: '0-rB52YBhEpkl2gfITkX',
        _score: 1.0,
        _source: {
          input: {
            tokens:
              'With a title \'borrowed\' from Werner Herzog and liberal helpings of Kubrick, Haneke and Noe it is painfully obvious that Thomas Clay considers himself a cut above the usual sort of rubbish our British cinema churns out. \'Robert Carmichael\' (for short) sets itself up as a realistic study of youthful alienation and at the same time seemingly a critique of the Iraq war. The problem with the realism is that the characters are so patently unrealistic and atypical - contrary to the fetid imaginings of \'extreme\' filmmakers most teenagers are not drug addled rapists. As a critique of the Iraq war, a film about youth violence (by a talented classical musician - subtext society has damaged this sensitive individual)is so infantile as to hardly bear thinking about. There are signs of technical ability but some reviewers have overstated this. Like Kubrick and Noe he does show that the desire to shock linked with supposed serious intent may be the worst cinematic con trick of recent film. People liked \'Clockwork Orange\' and \'Irreversible\' because they liked the rapes and the violence, but most of all they liked feeling culturally superior for liking things that most hated. So too much Kubrick and not enough Haneke (a serious and moral filmmaker) here labels this as one of the most moronic films in years. (I am not against violence in film. To do it seriously is a hard trick though - people in cinemas cheered Alex in \'Clockwork Orange\' showing how Kubrick\'s supposed intent was missed by miles. Gratuitous violence is much easier to achieve and is less offensive than the pretensions of many art-film directors.)',
            gold_label: 'negative',
            '@source': {
              label: 'negative',
              text:
                'With a title \'borrowed\' from Werner Herzog and liberal helpings of Kubrick, Haneke and Noe it is painfully obvious that Thomas Clay considers himself a cut above the usual sort of rubbish our British cinema churns out. \'Robert Carmichael\' (for short) sets itself up as a realistic study of youthful alienation and at the same time seemingly a critique of the Iraq war. The problem with the realism is that the characters are so patently unrealistic and atypical - contrary to the fetid imaginings of \'extreme\' filmmakers most teenagers are not drug addled rapists. As a critique of the Iraq war, a film about youth violence (by a talented classical musician - subtext society has damaged this sensitive individual)is so infantile as to hardly bear thinking about. There are signs of technical ability but some reviewers have overstated this. Like Kubrick and Noe he does show that the desire to shock linked with supposed serious intent may be the worst cinematic con trick of recent film. People liked \'Clockwork Orange\' and \'Irreversible\' because they liked the rapes and the violence, but most of all they liked feeling culturally superior for liking things that most hated. So too much Kubrick and not enough Haneke (a serious and moral filmmaker) here labels this as one of the most moronic films in years. (I am not against violence in film. To do it seriously is a hard trick though - people in cinemas cheered Alex in \'Clockwork Orange\' showing how Kubrick\'s supposed intent was missed by miles. Gratuitous violence is much easier to achieve and is less offensive than the pretensions of many art-film directors.)',
            },
          },
          prediction: {
            logits: [-0.591658890247345, 0.679482102394104],
            classes: {
              negative: 0.2190619856119156,
              positive: 0.7809380292892456,
            },
            max_class: 'positive',
            max_class_prob: 0.7809380292892456,
          },
          feedback: {
            prediction: {
              max_class: 'negative',
              max_class_prob: 1,
            },
            status: 'rejected',
          },
        },
        sort: [1.0, '0-rB52YBhEpkl2gfITkX'],
      },
      {
        _index: 'c490fa3a0fb8fdfe42586242ae9ae3b8',
        _type: 'docs',
        _id: '0OrB52YBhEpkl2gfITkX',
        _score: 1.0,
        _source: {
          input: {
            tokens:
              'Henri Verneuil represented the commercial cinema in France from 1960-1980. Always strong at the box-office, and usually telling dramatic and suspenseful tales of casino robberies, mafia score-settling and World War II battles, Verneuil could be counted on to give us two solid hours of entertainment on Saturday night. He worked with the cream of the male actors of his day: Gabin, Belmondo, Fernandel, Delon, Sharif, Anthony Quinn. I... comme Icare is the only time he directed Yves Montand. It\'s an oddly static film, taking place mainly in offices and conference rooms, containing not one chase scene and hardly any violence.<br /><br />Montand gives a good performance, if somewhat dry, and he is well supported by the other actors. I couldn\'t help wondering what Costa-Gavras could have done with this story, on the basis of Z (the Lambrakis assassination) and L\'aveu (the torture of Artur London in Czechoslovakia by Stalinists).',
            gold_label: 'positive',
            '@source': {
              label: 'positive',
              text:
                'Henri Verneuil represented the commercial cinema in France from 1960-1980. Always strong at the box-office, and usually telling dramatic and suspenseful tales of casino robberies, mafia score-settling and World War II battles, Verneuil could be counted on to give us two solid hours of entertainment on Saturday night. He worked with the cream of the male actors of his day: Gabin, Belmondo, Fernandel, Delon, Sharif, Anthony Quinn. I... comme Icare is the only time he directed Yves Montand. It\'s an oddly static film, taking place mainly in offices and conference rooms, containing not one chase scene and hardly any violence.<br /><br />Montand gives a good performance, if somewhat dry, and he is well supported by the other actors. I couldn\'t help wondering what Costa-Gavras could have done with this story, on the basis of Z (the Lambrakis assassination) and L\'aveu (the torture of Artur London in Czechoslovakia by Stalinists).',
            },
          },
          prediction: {
            logits: [-1.2658926248550415, 1.383965015411377],
            classes: {
              negative: 0.06599778681993484,
              positive: 0.9340022206306458,
            },
            max_class: 'positive',
            max_class_prob: 0.9340022206306458,
          },
          feedback: {
            prediction: {
              max_class: 'negative',
              max_class_prob: 1,
            },
            status: 'rejected',
          },
        },
        sort: [1.0, '0OrB52YBhEpkl2gfITkX'],
      },
      {
        _index: 'c490fa3a0fb8fdfe42586242ae9ae3b8',
        _type: 'docs',
        _id: '0erB52YBhEpkl2gfITkX',
        _score: 1.0,
        _source: {
          input: {
            tokens:
              'One of the most peculiar oft-used romance movie plots is this one: A seriously messed-up man falls in love with a terminally ill woman, who turns his life around before dying. Occasionally this story is done well and realistically (as in \'The Theory of Flight\', an excellent weepie), but more frequently it\'s done like it is here, where as usual the heroine dies of \'Old Movie Disease\'. You know, the terminal illness that has no symptoms but one fainting spell and a need to lie down as you\'re telling your lover goodbye forever; and your looks aren\'t affected one bit (and since this is the 70\'s, neither is your sex life). This is one of the worst versions made of that particular story, where a very silly script puts two incompatible and unbelievable characters together, and they\'re played by actors who are completely at sea.<br /><br />This has got to be the worst performance of Al Pacino\'s career, and I say that after having seen \'The Devil\'s Advocate\' only two days ago! He plays a control-freak, emotionally constipated race-car driver, and plays an unlikeable character lifelessly. He seems to constantly be asking himself why he\'s staying around the grating Marthe Keller (so does the audience), and spends most of the movie just... standing there, usually with his mouth hanging open. The only time he shows any sign of life is towards the end, where his character proves that he\'s changed from uptight to liberated by doing a hilariously bad Mae West imitation. Hey, it *was* the seventies!<br /><br />Marthe Keller is equally terrible as the dying love interest; her character was conceived as bold and free and touching and uninhibited and full of life even though dying, and was probably meant to be played with an actress with the sensitivity of, say, Vanessa Redgrave or Julie Christie. Instead, they got the expressionless face and heavy German accent of Ms. Keller, who comes across as more of a scary Teutonic stereotype (\'You VILL eat ze omelet!\') than anything like lovable. She\'s supposed to be reforming Pacino and filling him with courage and spirit and all that, but it doesn\'t work that way, it\'s more like she\'s harping on his faults in the most obnoxious possible fashion. This makes for one of the least convincing romances in movie history, where you can\'t believe she\'d be with someone she finds so worthless, and you can\'t believe ',
            gold_label: 'negative',
            '@source': {
              label: 'negative',
              text:
                'One of the most peculiar oft-used romance movie plots is this one: A seriously messed-up man falls in love with a terminally ill woman, who turns his life around before dying. Occasionally this story is done well and realistically (as in \'The Theory of Flight\', an excellent weepie), but more frequently it\'s done like it is here, where as usual the heroine dies of \'Old Movie Disease\'. You know, the terminal illness that has no symptoms but one fainting spell and a need to lie down as you\'re telling your lover goodbye forever; and your looks aren\'t affected one bit (and since this is the 70\'s, neither is your sex life). This is one of the worst versions made of that particular story, where a very silly script puts two incompatible and unbelievable characters together, and they\'re played by actors who are completely at sea.<br /><br />This has got to be the worst performance of Al Pacino\'s career, and I say that after having seen \'The Devil\'s Advocate\' only two days ago! He plays a control-freak, emotionally constipated race-car driver, and plays an unlikeable character lifelessly. He seems to constantly be asking himself why he\'s staying around the grating Marthe Keller (so does the audience), and spends most of the movie just... standing there, usually with his mouth hanging open. The only time he shows any sign of life is towards the end, where his character proves that he\'s changed from uptight to liberated by doing a hilariously bad Mae West imitation. Hey, it *was* the seventies!<br /><br />Marthe Keller is equally terrible as the dying love interest; her character was conceived as bold and free and touching and uninhibited and full of life even though dying, and was probably meant to be played with an actress with the sensitivity of, say, Vanessa Redgrave or Julie Christie. Instead, they got the expressionless face and heavy German accent of Ms. Keller, who comes across as more of a scary Teutonic stereotype (\'You VILL eat ze omelet!\') than anything like lovable. She\'s supposed to be reforming Pacino and filling him with courage and spirit and all that, but it doesn\'t work that way, it\'s more like she\'s harping on his faults in the most obnoxious possible fashion. This makes for one of the least convincing romances in movie history, where you can\'t believe she\'d be with someone she finds so worthless, and you can\'t believe he',
            },
          },
          prediction: {
            logits: [0.539858341217041, -0.4038154184818268],
            classes: {
              negative: 0.7198411226272583,
              positive: 0.2801588773727417,
            },
            max_class: 'negative',
            max_class_prob: 0.7198411226272583,
          },
          feedback: {
            prediction: {
              max_class: 'positive',
              max_class_prob: 1,
            },
            status: 'rejected',
          },
        },
        sort: [1.0, '0erB52YBhEpkl2gfITkX'],
      },
      {
        _index: 'c490fa3a0fb8fdfe42586242ae9ae3b8',
        _type: 'docs',
        _id: '0urB52YBhEpkl2gfITkX',
        _score: 1.0,
        _source: {
          input: {
            tokens:
              'I had suspicions the movie was going to be bad. ',
            gold_label: 'negative',
            '@source': {
              label: 'negative',
              text:
                'I had suspicions the movie was going to be bad. ',
            },
          },
          prediction: {
            logits: [-0.11335738003253937, 0.024933991953730583],
            classes: {
              negative: 0.46548211574554443,
              positive: 0.5345178246498108,
            },
            max_class: 'positive',
            max_class_prob: 0.5345178246498108,
          },
          feedback: {
            prediction: {
              max_class: 'negative',
              max_class_prob: 1,
            },
            status: 'rejected',
          },
        },
        sort: [1.0, '0urB52YBhEpkl2gfITkX'],
      },
      {
        _index: 'c490fa3a0fb8fdfe42586242ae9ae3b8',
        _type: 'docs',
        _id: '1-rB52YBhEpkl2gfITkX',
        _score: 1.0,
        _source: {
          input: {
            tokens:
              'This movie is horrible- in a ',
            gold_label: 'positive',
            '@source': {
              label: 'positive',
              text:
                'This movie is horrible- in a ',
            },
          },
          prediction: {
            logits: [-0.36276882886886597, 0.3100770115852356],
            classes: {
              negative: 0.3378598988056183,
              positive: 0.6621401309967041,
            },
            max_class: 'positive',
            max_class_prob: 0.6621401309967041,
          },
          feedback: {
            prediction: {
              max_class: 'negative',
              max_class_prob: 1,
            },
            status: 'rejected',
          },
        },
        sort: [1.0, '1-rB52YBhEpkl2gfITkX'],
      },
      {
        _index: 'c490fa3a0fb8fdfe42586242ae9ae3b8',
        _type: 'docs',
        _id: '1OrB52YBhEpkl2gfITkX',
        _score: 1.0,
        _source: {
          input: {
            tokens:
              'I am starting this review with a big giant spoiler about this film. Do not read further...here it comes, avert your eyes! The main heroine, the girl who always survives in other slasher films, is murdered here. There, I just saved you 79 minutes of your life.<br /><br />This is one of those cheap movies that was thrown together in the middle of the slasher era of the ',
            gold_label: 'negative',
            '@source': {
              label: 'negative',
              text:
                'I am starting this review with a big giant spoiler about this film. Do not read further...here it comes, avert your eyes! The main heroine, the girl who always survives in other slasher films, is murdered here. There, I just saved you 79 minutes of your life.<br /><br />This is one of those cheap movies that was thrown together in the middle of the slasher era of the ',
            },
          },
          prediction: {
            logits: [-0.014385417103767395, -0.06117277592420578],
            classes: {
              negative: 0.5116947293281555,
              positive: 0.48830530047416687,
            },
            max_class: 'negative',
            max_class_prob: 0.5116947293281555,
          },
          feedback: {
            prediction: {
              max_class: 'positive',
              max_class_prob: 1,
            },
            status: 'rejected',
          },
        },
        sort: [1.0, '1OrB52YBhEpkl2gfITkX'],
      },
    ],
  },
  aggregations: {
    gold: {
      doc_count_error_upper_bound: 0,
      sum_other_doc_count: 0,
      buckets: [
        {
          key: 'negative',
          doc_count: 107,
        },
        {
          key: 'positive',
          doc_count: 93,
        },
      ],
    },
    confidence_range: {
      buckets: [
        {
          key: '0.0-0.05',
          from: 0.0,
          to: 0.05,
          doc_count: 0,
        },
        {
          key: '0.05-0.1',
          from: 0.05,
          to: 0.1,
          doc_count: 0,
        },
        {
          key: '0.1-0.15000000000000002',
          from: 0.1,
          to: 0.15000000000000002,
          doc_count: 0,
        },
        {
          key: '0.15000000000000002-0.2',
          from: 0.15000000000000002,
          to: 0.2,
          doc_count: 0,
        },
        {
          key: '0.2-0.25',
          from: 0.2,
          to: 0.25,
          doc_count: 0,
        },
        {
          key: '0.25-0.3',
          from: 0.25,
          to: 0.3,
          doc_count: 0,
        },
        {
          key: '0.30000000000000004-0.35000000000000003',
          from: 0.30000000000000004,
          to: 0.35000000000000003,
          doc_count: 0,
        },
        {
          key: '0.35000000000000003-0.4',
          from: 0.35000000000000003,
          to: 0.4,
          doc_count: 0,
        },
        {
          key: '0.4-0.45',
          from: 0.4,
          to: 0.45,
          doc_count: 0,
        },
        {
          key: '0.45-0.5',
          from: 0.45,
          to: 0.5,
          doc_count: 0,
        },
        {
          key: '0.5-0.55',
          from: 0.5,
          to: 0.55,
          doc_count: 22,
        },
        {
          key: '0.55-0.6000000000000001',
          from: 0.55,
          to: 0.6000000000000001,
          doc_count: 15,
        },
        {
          key: '0.6000000000000001-0.6500000000000001',
          from: 0.6000000000000001,
          to: 0.6500000000000001,
          doc_count: 16,
        },
        {
          key: '0.65-0.7000000000000001',
          from: 0.65,
          to: 0.7000000000000001,
          doc_count: 20,
        },
        {
          key: '0.7000000000000001-0.7500000000000001',
          from: 0.7000000000000001,
          to: 0.7500000000000001,
          doc_count: 19,
        },
        {
          key: '0.75-0.8',
          from: 0.75,
          to: 0.8,
          doc_count: 24,
        },
        {
          key: '0.8-0.8500000000000001',
          from: 0.8,
          to: 0.8500000000000001,
          doc_count: 20,
        },
        {
          key: '0.8500000000000001-0.9000000000000001',
          from: 0.8500000000000001,
          to: 0.9000000000000001,
          doc_count: 29,
        },
        {
          key: '0.9-0.9500000000000001',
          from: 0.9,
          to: 0.9500000000000001,
          doc_count: 30,
        },
      ],
    },
    confusion_matrix: {
      doc_count_error_upper_bound: 0,
      sum_other_doc_count: 0,
      buckets: [
        {
          key: 'negative',
          doc_count: 107,
          predicted: {
            doc_count_error_upper_bound: 0,
            sum_other_doc_count: 0,
            buckets: [
              {
                key: 'positive',
                doc_count: 55,
              },
              {
                key: 'negative',
                doc_count: 52,
              },
            ],
          },
        },
        {
          key: 'positive',
          doc_count: 93,
          predicted: {
            doc_count_error_upper_bound: 0,
            sum_other_doc_count: 0,
            buckets: [
              {
                key: 'positive',
                doc_count: 78,
              },
              {
                key: 'negative',
                doc_count: 15,
              },
            ],
          },
        },
      ],
    },
    predicted: {
      doc_count_error_upper_bound: 0,
      sum_other_doc_count: 0,
      buckets: [
        {
          key: 'positive',
          doc_count: 133,
        },
        {
          key: 'negative',
          doc_count: 67,
        },
      ],
    },
  },
};
