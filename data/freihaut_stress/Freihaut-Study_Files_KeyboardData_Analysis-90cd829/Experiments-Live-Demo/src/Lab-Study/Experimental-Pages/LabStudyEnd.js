// final page of experiment

import React, { Component } from 'react';


export default class LabStudyEnd extends Component {

    componentDidMount() {
        window.scrollTo(0, 0);
    }

    render() {
        return(
            <div className="section">
                <div className="content" style={{}}>
                    <h3>You finished the study. Thank you very much for your participation!</h3><br/><br/>
                    <div style={{textAlign: "left"}}>
                        <h5>Final information</h5>
                        <p>
                            Your performance was not monitored and evaluated during the exercise block. The purpose of the false announcement was to increase your stress level during the tasks. The procedure was necessary for our research question. The goal of the study is to investigate an effect of stress on computer mouse and keyboard usage.
                        </p>
                        <p>
                            If the study is still stressing you, please remain seated and take your time to relax.
                        </p>
                        <p>
                            If you have final questions or feedback regarding this study, please contact the experimenter or write an e-mail.
                        </p>
                    </div>
                </div>
            </div>
        );
    }

}