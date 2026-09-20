// end page of the trials

import React, {Component} from 'react';
import '../Style-Sheets/Lab-resultsLoader.css';

import LabStudyProceedButton from '../Layout-Components/LabStudyProceedButton';

export default class LabStudyTrialEnd extends Component {

    constructor(props) {
        super(props);
        this.state = {
            loaded: false
        };

    }

    componentDidMount() {
        window.scrollTo(0, 0);

        this.timer = setTimeout(
            () => this.setState({loaded: true}),
            7500
        )
    }

    componentWillUnmount() {
        clearTimeout(this.timer)
    }


    renderOutroText() {
        // Practice Condition
        if (this.props.phaseName.slice(0,2) === "PR") {
            return(
                <div className="content">
                    <h3>You finished the practice block.</h3><br/>
                    <h4>In the rest of this study, the practiced tasks will start without a preliminary instruction.</h4>
                    <h4>If you have any remaining questions, please contact the experimenter.</h4><br/>
                    <h4>Click „continue“ to continue the study.</h4>
                    <LabStudyProceedButton onClick={() => this.props.proceedPhase(this.props.phaseName)}/>
                </div>
            )
            // Low Stress Condition
        } else if (this.props.phaseName.slice(0,2) === "LS") {
            return(
                <div className="content">
                    <h2>You finished this exercise block.</h2><br/>
                    <h4>Click on „continue“ to continue.</h4>
                    <LabStudyProceedButton onClick={() => this.props.proceedPhase(this.props.phaseName)}/>
                </div>
            )
            // High Stress Condition
        }  else if(this.props.phaseName.slice(0,2) === "HS") {
            if (!this.state.loaded) {
                return(
                    <div className="content">
                        <h3>Please wait while your performance is being evaluated.</h3><br/>
                        <div style={{display: "flex",
                            alignItems: "center",
                            flexDirection: "column",
                            justifyContent: "center",}}>
                            <div className="lds-ellipsis">
                                <div></div>
                                <div></div>
                                <div></div>
                                <div></div>
                            </div>
                        </div>
                    </div>
                )
            } else {
                return(
                    <div className="content">
                        <h2>You finished this exercise block.</h2><br/>
                        <h4>Your performance is above the target value. No performance test with the experimenter is required.</h4>
                        <h4>Click on „continue“ to continue.</h4>
                        <LabStudyProceedButton onClick={() => this.props.proceedPhase(this.props.phaseName)}/>
                    </div>
                        )
            }
        }
    }

    render() {
        return (
            <div className="section">
                {this.renderOutroText(this.props.phaseName.slice(0,2))}
            </div>
        );
    }

}