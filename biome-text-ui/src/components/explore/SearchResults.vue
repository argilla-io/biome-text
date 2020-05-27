<template>
  <div class="results" id="results" ref="resultsList">
    <div class="list list--explore">
      <transition name="fade" appear v-if="records">
        <div>
          <loading-skeleton :condition="loadingQ" :items-number="10" :buttons-number="null"></loading-skeleton>
          <div>
            <div class="help">
              <div class="help__button" v-if="!showHelpPanel" @click="showHelpPanel = true">
                <svgicon
                  name="help"
                  width="22"
                  height="22"
                  color="#F38959"
                ></svgicon>
                 Help
              </div>
              <div class="help__panel" v-if="showHelpPanel">
                <div class="help__panel__button" @click="showHelpPanel = false">
                  close
                </div>
                <p class="help__panel__title" >How is learning the model?</p>
                <p><span class="atom grad-neg-30"><span>Eddie</span></span> from -1 to 0</p>
                <p><span class="atom grad-30"><span>Murphy</span></span> from  0 to 1</p>
              </div>
            </div>
            <div v-for="(item, key) in records" :key="item._id" class="list__li">
              <div
                class="list__item"
                v-waypoint="{ active: true, callback: onWaypoint,  options: checkPageNumber}"
              >
                <a v-if="scrollerPosition > 100 && key >= 18" class="page__number__scroll-button" href="#app" v-smooth-scroll="{ duration: 1000, offset: -50 }"></a>
                <div class="pill__container">
                  <p class="pill--gold" :data-title="item.gold">{{item.gold}}</p>
                  <div class="pill--predicted__container">
                    <re-numeric :value="decorateConfidence(item.confidence)" type="%" :decimals="2"></re-numeric>
                    <p
                      class="pill--predicted"
                      :class="[{'ko' : !item.predictedOk, 'neutral' : !item.gold.length}]"
                      :data-title="item.predicted"
                    >
                      {{item.predicted }}
                    </p>
                  </div>
                </div>
                <record
                  :query="query"
                  :showEntityClassifier="showEntityClassifier"
                  :explain="item.interpretations"
                  :record="item.inputs"
                  :predictedOk="item.predictedOk"
                ></record>
                <div class="list__extra-actions">
                  <div
                    v-if="Object.values(item.metadata).length"
                    @click="showMetadata = item._id"
                  >View metadata</div>
                </div>
              </div>
              <re-modal
                :modalCustom="true"
                :preventBodyScroll="true"
                modalClass="modal-classifier"
                :modalVisible="showMetadata === item._id"
                modalPosition="modal-center"
                @close-modal="closeModal()"
              >
                <metadata
                  :filtersStatus="filtersStatus"
                  :metaFiltrable="metaFiltrable"
                  :previewData="item.metadata"
                  :inputs="item.inputs"
                  @metafilterapply="onMetaFilterApply"
                  @cancel="closeModal()"
                ></metadata>
              </re-modal>
            </div>
            <div class="show-more-data" v-if="records.length && records.length < total">
              <p>{{records.length}} of {{total}} records</p>
              <re-progress
                class="re-progress--minimal"
                re-mode="determinate"
                :progress="shownRecordsProgress"
              ></re-progress>
              <re-button class="button-tertiary--outline" @click="moreData()">Next 20 records</re-button>
            </div>
          </div>
        </div>
      </transition>
    </div>
  </div>
</template>
<script>
import searchResultsMixin from '../mixins/MixinSearchResults';
import '@/assets/iconsfont/help';
import '@/assets/iconsfont/cross';

export default {
  name: 'search-results',
  data: () => ({
    showHelpPanel: false,
  }),
  mixins: [searchResultsMixin],
};
</script>
<style lang="scss" scoped>
@import "@/assets/scss/components/list.scss";
@import "@/assets/scss/components/help.scss";
@import "@/assets/scss/components/tooltip.scss";
@import "@/assets/scss/apps/classifier/show-more-data.scss";
@import "@/assets/scss/apps/classifier/explore.scss";
@import "@/assets/scss/apps/classifier/explain.scss";
</style>
